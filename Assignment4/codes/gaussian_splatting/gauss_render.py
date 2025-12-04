import pdb
import torch
import torch.nn as nn
import math
from einops import reduce

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R



def build_scaling_rotation(s, r):
    '''
    Args:
    s (torch:Tensor): scale of gaussian with shape (N, 3)
    r (torch:Tensor): quaternion of gaussian with shape (N, 4)

    Using the provided build_rotation to get the rotation matrix from quaternion and build scale matrix for s.
    
    Return:
    (torch:Tensor) the matrix production of rotation matrix and scale matrix with shape (N, 3, 3) 
    '''
    # ensure tensors are float and on same device
    s = s.float()
    r = r.float()
    device = r.device if r is not None else s.device

    # build rotation matrix R from quaternion r: (N,3,3)
    R = build_rotation(r)  # build_rotation already returns a cuda tensor in your code

    # build scale matrix S = diag(sx, sy, sz) for each gaussian: (N,3,3)
    # preserve device
    S = torch.zeros((s.shape[0], 3, 3), device=device, dtype=s.dtype)
    S[:, 0, 0] = s[:, 0]
    S[:, 1, 1] = s[:, 1]
    S[:, 2, 2] = s[:, 2]

    # return R @ S  (RS). Later covariance = (RS)(RS)^T
    RS = torch.matmul(R, S)
    return RS


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)



def corvariance_3d(s, r):
    '''
    Args:
        s (torch:Tensor): scale of gaussian with shape (N, 3)
        r (torch:Tensor): quaternion of gaussian with shape (N, 4)
        
        We use build_scaling_rotation to build matrix of R S and then we can obtain the covariance of 3D gaussian by RS(RS)^T
    
    Return
        (torch:Tensor)) 3D covariance with shape (N, 3, 3)
    '''
    # build RS (N,3,3)
    RS = build_scaling_rotation(s, r)
    # covariance = RS @ RS^T  => yields R * S * S * R^T (i.e., scales squared)
    cov3d = torch.matmul(RS, RS.transpose(-1, -2))
    return cov3d


def corvariance_2d(
    mean3d, cov3d, viewmatrix, fov_x, fov_y, focal_x, focal_y
):
    # print("DEBUG cov3d shape:", cov3d.shape)
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_y * 0.5)

    # -----------------------------------------------------
    # (1) Transform 3D means from world → camera coordinates
    # -----------------------------------------------------
    # viewmatrix: (4,4)
    # rotation = upper-left 3x3
    # translation = the LAST column, not last row!
    R = viewmatrix[:3, :3]          # (3,3)
    T = viewmatrix[:3, 3]           # (3,)

    # mean3d: (N,3)
    # t = R * x + T
    t = mean3d @ R.T + T[None, :]    # (N,3)


    # -----------------------------------------------------
    # (2) Frustum clipping for stability (EWQ paper trick)
    # -----------------------------------------------------
    tx = (t[..., 0] / t[..., 2]).clip(min=-tan_fovx * 1.3, max=tan_fovx * 1.3) * t[..., 2]
    ty = (t[..., 1] / t[..., 2]).clip(min=-tan_fovy * 1.3, max=tan_fovy * 1.3) * t[..., 2]
    tz = t[..., 2]

    # -----------------------------------------------------
    # (3) Build the Jacobian J for projection
    # -----------------------------------------------------
    # Projection: x_img = focal_x * (tx / tz)
    #             y_img = focal_y * (ty / tz)
    #
    # J = d[x_img, y_img] / d[tx, ty, tz]
    #
    # Each Gaussian has its own J: (N, 2, 3)
    eps = 1e-6
    inv_z = 1.0 / (tz + eps)

    J = torch.zeros((mean3d.shape[0], 2, 3), device=mean3d.device)

    # d(x_img)/d(tx) = focal_x / tz
    J[:, 0, 0] = focal_x * inv_z
    # d(x_img)/d(ty) = 0
    # d(x_img)/d(tz) = -focal_x * tx / tz^2
    J[:, 0, 2] = -focal_x * tx * inv_z * inv_z

    # d(y_img)/d(tx) = 0
    # d(y_img)/d(ty) = focal_y / tz
    J[:, 1, 1] = focal_y * inv_z
    # d(y_img)/d(tz) = -focal_y * ty / tz^2
    J[:, 1, 2] = -focal_y * ty * inv_z * inv_z

    
    # -----------------------------------------------------
    # (4) Rotate 3D covariance into camera coordinates
    #     covCam = W * cov3d * W^T
    # -----------------------------------------------------
    W = R.T  # (3,3) world -> camera rotation

    # broadcast W to batch: (1,3,3) so we can multiply with cov3d (N,3,3)
    W_b = W.unsqueeze(0)  # (1,3,3)
    # cov_cam = W_b @ cov3d @ W_b.transpose(-1, -2)  # (N,3,3)
    # to be explicit and safe with broadcasting:
    cov_cam = torch.matmul(W_b, cov3d)            # (N,3,3)
    cov_cam = torch.matmul(cov_cam, W_b.transpose(-1, -2))  # (N,3,3)

    # -----------------------------------------------------
    # (5) 2D covariance = J * cov_cam * J^T
    # -----------------------------------------------------
    cov2d = J @ cov_cam @ J.transpose(1, 2)  # (N, 2, 2)

    # -----------------------------------------------------
    # (6) Add EWQ low-pass filter (Eq. 32)
    # -----------------------------------------------------
    filter = torch.eye(2, 2, device=cov2d.device) * 0.3
    return cov2d + filter[None]



def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= -10
    return p_proj, p_view, in_mask


@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


from .utils.sh_utils import eval_sh

class GaussRenderer(nn.Module):

    def __init__(self, active_sh_degree=3, white_bkgd=True, pixel_range=256, **kwargs):
        super(GaussRenderer, self).__init__()
        self.active_sh_degree = active_sh_degree
        self.debug = False
        self.white_bkgd = white_bkgd
        self.pix_coord = torch.stack(torch.meshgrid(torch.arange(pixel_range), torch.arange(pixel_range), indexing='xy'), dim=-1).to('cuda')
              
    
    def build_color(self, means3D, shs, camera):
        rays_o = camera.camera_center
        rays_d = means3D - rays_o
        color = eval_sh(self.active_sh_degree, shs.permute(0,2,1), rays_d)
        color = (color + 0.5).clip(min=0.0)
        return color
    
    def render(self, camera, means2D, cov2d, color, opacity, depths):
        
        """
        Tile-based 2D Gaussian splatting renderer.

        Args:
            camera:
                Camera object providing image resolution:
                    - camera.image_width  (int)
                    - camera.image_height (int)
                self.pix_coord is assumed to be a precomputed (H, W, 2) tensor of pixel coords.

            means2D (torch.Tensor): (N, 2)
                Projected 2D centers of Gaussians in image space (pixel coordinates).

            cov2d (torch.Tensor): (N, 2, 2)
                2D image-space covariance matrices of Gaussians.

            color (torch.Tensor): (N, 3)
                RGB color for each Gaussian (in [0, 1]).

            opacity (torch.Tensor): (N, 1)
                Per-Gaussian base opacity (before per-pixel Gaussian weighting).

            depths (torch.Tensor): (N,)
                Per-Gaussian depth values (e.g., z in camera space),
                used for front-to-back sorting when compositing.
        Returns:
            dict with:
                - "render": (H, W, 3) final RGB image.
                - "alpha":  (H, W, 1) accumulated alpha map.
                - "visiility_filter": (N,) visibility mask (radii > 0).
                - "radii": (N,) per-Gaussian screen-space radius.
        """
        radii = get_radius(cov2d)
        rect = get_rect(means2D, radii, width=camera.image_width, height=camera.image_height)
        
        self.render_color = torch.ones(*self.pix_coord.shape[:2], 3).to('cuda')
        self.render_alpha = torch.zeros(*self.pix_coord.shape[:2], 1).to('cuda')

        TILE_SIZE = 25
        H = camera.image_height
        W = camera.image_width
        N = means2D.shape[0]
        device = means2D.device

        for h in range(0, H, TILE_SIZE):
            for w in range(0, W, TILE_SIZE):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+TILE_SIZE-1), rect[1][..., 1].clip(max=h+TILE_SIZE-1)
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1]) # 3D gaussian in the tile 
                
                if not in_mask.sum() > 0:
                    continue

                # Extract the pixel coordinates for this tile.
                tile_h = min(TILE_SIZE, H - h)
                tile_w = min(TILE_SIZE, W - w)
                # get pixel coords in the tile and flatten: (B,2)
                tile_coord = self.pix_coord[h:h+tile_h, w:w+tile_w].reshape(-1, 2).float().to(device)  # (B,2)
                B = tile_coord.shape[0]

                # Sort Gaussians by depth (near to far)
                inds = torch.nonzero(in_mask).squeeze(-1)  # indices of gaussians relevant to this tile
                local_depths = depths[inds]
                sorted_idx_local = torch.argsort(local_depths)  # ascending: near->far
                sorted_inds = inds[sorted_idx_local]

                # Extract relevant Gaussian properties for the tile.
                sorted_means2D = means2D[sorted_inds]          # (M,2)
                sorted_cov2d = cov2d[sorted_inds]              # (M,2,2)
                sorted_opacity = opacity[sorted_inds].reshape(-1)    # (M,)
                sorted_color = color[sorted_inds]              # (M,3)
                M = sorted_means2D.shape[0]

                if M == 0:
                    continue

                # Compute the difference between each pixel and gaussian centers: (B, M, 2)
                # tile_coord: (B,2) -> (B,1,2); means -> (1,M,2)
                diff = tile_coord[:, None, :] - sorted_means2D[None, :, :]  # (B, M, 2)

                # Compute Gaussian weights using Mahalanobis distance
                # Precompute inverse covariances for each gaussian (M,2,2)
                # Add small eps for numerical stability
                eps_eye = 1e-6 * torch.eye(2, device=device)[None, :, :]
                Sigma_inv = torch.linalg.inv(sorted_cov2d + eps_eye)  # (M,2,2)

                # compute tmp = diff @ Sigma_inv for each gaussian (B,M,2)
                # use einsum for clarity
                tmp = torch.einsum('bmd,mdk->bmk', diff, Sigma_inv)  # (B, M, 2)
                exponent = -0.5 * torch.sum(tmp * diff, dim=-1)  # (B, M)
                gauss_weight = torch.exp(exponent)  # (B, M)

                # per-pixel alpha for each gaussian: alpha = tau * gauss_weight
                alpha_per_pixel = (sorted_opacity[None, :] * gauss_weight)  # (B, M)

                # Compute transmittance T and weights for front-to-back compositing
                # For each pixel b, compute cumulative product over axis M
                one = torch.ones((B, 1), device=device)
                cumprod_input = torch.cat([one, 1.0 - alpha_per_pixel + 1e-10], dim=1)  # (B, M+1)
                T = torch.cumprod(cumprod_input, dim=1)[:, :-1]  # (B, M) transmittance before each gaussian
                weights = T * alpha_per_pixel  # (B, M)

                # Weighted color and alpha accumulation per pixel
                # sorted_color: (M,3) -> broadcast to (B,M,3)
                weighted_colors = weights[:, :, None] * sorted_color[None, :, :]  # (B, M, 3)
                tile_color = torch.sum(weighted_colors, dim=1)  # (B, 3)
                tile_alpha = torch.sum(weights, dim=1, keepdim=True)  # (B,1)

                # Write tile results to full buffers: reshape back to (tile_h, tile_w, ...)
                tile_color_img = tile_color.reshape(tile_h, tile_w, 3)
                tile_alpha_img = tile_alpha.reshape(tile_h, tile_w, 1)

                self.render_color[h:h+tile_h, w:w+tile_w] = tile_color_img
                self.render_alpha[h:h+tile_h, w:w+tile_w] = tile_alpha_img


        return {
            "render": self.render_color,
            "alpha": self.render_alpha,
            "visiility_filter": radii > 0,
            "radii": radii
        }



    def forward(self, camera, pc, **kwargs):
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
        

        mean_ndc, mean_view, in_mask = projection_ndc(means3D, 
                viewmatrix=camera.world_view_transform, 
                projmatrix=camera.projection_matrix)
        mean_ndc = mean_ndc  # [in_mask]
        mean_view = mean_view # [in_mask]
        depths = mean_view[:,2]
        
        color = self.build_color(means3D=means3D, shs=shs, camera=camera)
        
        cov3d = corvariance_3d(scales, rotations)
            
        cov2d = corvariance_2d(
            mean3d=means3D, 
            cov3d=cov3d, 
            viewmatrix=camera.world_view_transform,
            fov_x=camera.FoVx, 
            fov_y=camera.FoVy, 
            focal_x=camera.focal_x, 
            focal_y=camera.focal_y)

        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
    
        rets = self.render(
            camera = camera, 
            means2D=means2D,
            cov2d=cov2d,
            color=color,
            opacity=opacity, 
            depths=depths,
        )

        return rets
