import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr

from nerf_dataset import NerfDataset
from nerf_model import NeRF

################################ IMPORTANT: This model is quite slow, you do not need to run it until it converges.  ###################################

# Position Encoding
class PositionalEncoder(nn.Module):
    """
    Implement the Position Encoding function.
    Defines a function that embeds x to (sin(2^k*pi*x), cos(2^k*pi*x), ...)
    Please note that the input tensor x should be normalized to the range [-1, 1].

    Args:
    x (torch.Tensor): The input tensor to be embedded.
    L (int): The number of levels to embed.

    Returns:
    torch.Tensor: The embedded tensor.
    """
    def __init__(self, data_range, L, include_input=False):
        super(PositionalEncoder, self).__init__()
        self.L = L
        self.include_input = include_input
        self.min_val = data_range[0]
        self.max_val = data_range[1]

        # frequency bands: 1,2,4,...,2^(L-1)
        self.freq_bands = 2 ** torch.arange(L)

    def forward(self, x):
        # normalize x into [-1,1]
        x_norm = 2 * (x - self.min_val) / (self.max_val - self.min_val) - 1

        encoding = []

        # include raw input or not
        if self.include_input:
            encoding.append(x_norm)

        # add sin/cos terms
        for freq in self.freq_bands:
            encoding.append(torch.sin(freq * torch.pi * x_norm))
            encoding.append(torch.cos(freq * torch.pi * x_norm))

        return torch.cat(encoding, dim=-1)


def sample_rays(H, W, f, c2w):
    """
    Samples rays from a camera with given height H, width W, focal length f, and camera-to-world matrix c2w.

    Args:
    H (int): The height of the image.
    W (int): The width of the image.
    f (float): The focal length of the camera.
    c2w (torch.Tensor): The 4x4 camera-to-world transformation matrix.

    Returns:
    rays_o (torch.Tensor): The origin of each ray, with shape (W, H, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (W, H, 3).
    """
    device = c2w.device

    # Create grid of pixel coordinates
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='ij'
    )

    # Convert pixel coordinates to camera coordinates
    x_cam = (i - W * 0.5) / f
    y_cam = -(j - H * 0.5) / f
    z_cam = -torch.ones_like(x_cam, device=device)

    dirs = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # (W,H,3)

    # Rotate ray directions
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    # Ray origins = camera position
    rays_o = c2w[:3, 3].expand_as(rays_d)

    return rays_o, rays_d

def sample_points_along_the_ray(tn, tf, N_samples):
    """
    Samples points uniformly along a ray from time t_n to time t_f.

    Args:
    tn (torch.Tensor): The starting point of the ray.
    tf (torch.Tensor): The ending point of the ray.
    N_samples (int): The number of samples to take along the ray.

    Returns:
    torch.Tensor: A tensor of shape (N_samples, ...) containing the sampled points along the ray,
                  where ... corresponds to the shape of tn or tf.
    """
    # Generate uniform bin edges (N_samples+1)
    # e.g. [0, 1/N, 2/N, ..., 1]
    bins = torch.linspace(0.0, 1.0, N_samples + 1, device=tn.device)

    # Uniform samples inside each bin (stratified)
    # random jitter in [0, 1/N]
    jitter = torch.rand(*tn.shape[:-1], N_samples, device=tn.device)

    # Convert bins to actual sample locations in [tn, tf]
    # t = tn + (bins[:-1] + jitter*(1/N_samples)) * (tf - tn)
    t_vals = tn + (bins[:-1] + jitter / N_samples) * (tf - tn)

    return t_vals

def volumn_render(NeRF, rays_o, rays_d, N_samples):
    """
    Performs volume rendering to generate an image from rays.

    Args:
    NeRF (nn.Module): The neural radiance field model.
    rays_o (torch.Tensor): The origin of each ray, with shape (N_rays, 3).
    rays_d (torch.Tensor): The direction of each ray, with shape (N_rays, 3).
    N_samples (int): The number of samples to take along each ray.

    Returns:
    torch.Tensor: The rendered RGB image.
    """
    device = rays_o.device
    N_rays = rays_o.shape[0]

    # Near plane and far plane
    tn = torch.full((N_rays, 1), 2.0, device=device)   # you may adjust these
    tf = torch.full((N_rays, 1), 6.0, device=device)

    # Sample t values: (N_rays, N_samples)
    t_vals = sample_points_along_the_ray(tn, tf, N_samples)

    # Compute sampled 3D points
    # pts = o + t * d
    pts = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None]
    # pts shape: (N_rays, N_samples, 3)

    # Flatten for NeRF MLP: (N_rays*N_samples, 3)
    pts_flat = pts.reshape(-1, 3)

    # directions need to be repeated for each sample
    # (N_rays, 3) -> (N_rays, N_samples, 3)
    dirs = rays_d[:, None, :].expand_as(pts)
    dirs_flat = dirs.reshape(-1, 3)

    # Query NeRF
    # NeRF returns (rgb, sigma)
    rgb, sigma = NeRF(pts_flat, dirs_flat)
    rgb = rgb.reshape(N_rays, N_samples, 3)
    sigma = sigma.reshape(N_rays, N_samples)

    # Compute distances between adjacent samples
    # delta_i = t[i+1] - t[i]
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    deltas = torch.cat([deltas, torch.full((N_rays, 1), 1e10, device=device)], dim=-1)

    # Convert density to alpha
    # alpha_i = 1 - exp(-sigma_i * delta_i)
    alpha = 1.0 - torch.exp(-torch.nn.functional.softplus(sigma) * deltas)

    # Compute transmittance T_i = Π_{j<i} (1 - alpha_j)
    T = torch.cumprod(torch.cat([torch.ones((N_rays, 1), device=device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]

    # Weight for each sample
    weights = T * alpha  # (N_rays, N_samples)

    # Final RGB = Σ_i weights_i * rgb_i
    rgb_map = torch.sum(weights[..., None] * rgb, dim=1)

    return rgb_map
    


def random_select_rays(H, W, rays_o, rays_d, img, N_rand):
    """
    Randomly select N_rand rays to reduce memory usage.

    Parameters:
    - H: int, height of the image.
    - W: int, width of the image.
    - rays_o: torch.Tensor, original ray origins with shape (H * W, 3).
    - rays_d: torch.Tensor, ray directions with shape (H * W, 3).
    - img: torch.Tensor, image with shape (H * W, 3).
    - N_rand: int, number of random rays to select.

    Returns:
    - selected_rays_o: torch.Tensor, selected ray origins with shape (N_rand, 3).
    - selected_rays_d: torch.Tensor, selected ray directions with shape (N_rand, 3).
    - selected_img: torch.Tensor, selected image pixels with shape (N_rand, 3).
    """
    # Generate coordinates for all pixels in the image
    coords = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), -1)  # (H, W, 2)
    coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
    
    # Randomly select N_rand indices without replacement
    select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)
    
    # Select the corresponding coordinates, rays, and image pixels
    select_coords = coords[select_inds].long().to("cpu")  # (N_rand, 2)
    selected_rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = img[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    selected_img = torch.tensor(selected_img, dtype=torch.float32)  # Ensure float32 dtype
    
    return selected_rays_o, selected_rays_d, selected_img


def fit_images_and_calculate_psnr(data_path, epochs=2000, learning_rate=5e-4):
    # get available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data
    dataset = NerfDataset(data_path)

    # create model
    xyz_encoder = PositionalEncoder(data_range=[-4, 4], L=10).to(device)
    dir_encoder = PositionalEncoder(data_range=[-1, 1], L=4).to(device)
    nerf = NeRF(
        xyz_encoder=xyz_encoder,
        dir_encoder=dir_encoder,
        input_dim=60,
        view_dim=24,
    ).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=learning_rate)
    loss = nn.MSELoss()

    # train the model
    N_samples = 64 # number of samples per ray
    N_rand = 1024 # number of rays per iteration, adjust according to your GPU memory
    for epoch in tqdm(range(epochs+1)):
        for i in range(len(dataset)):
            img, pose, focal = dataset[i]
            img = img.to(device)
            H, W = img.shape[:2]
            pose = pose.to(device)
            focal = focal.to(device)

            # sample rays
            rays_o, rays_d = sample_rays(H, W, focal, c2w=pose)

            # random select N_rand rays to reduce memory usage
            selected_rays_o, selected_rays_d, selected_gt_rgb = random_select_rays(H, W, rays_o, rays_d, img, N_rand)

            # volumn render
            pred_rgb = volumn_render(NeRF=nerf, rays_o=selected_rays_o, rays_d=selected_rays_d, N_samples=N_samples)

            l = loss(pred_rgb, selected_gt_rgb)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            psnr_value = psnr(selected_gt_rgb.detach().cpu().numpy(), pred_rgb.detach().cpu().numpy(), data_range=1)

        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {l.item()}, PSNR: {psnr_value}')
            with torch.no_grad():
                chunk_size = 1024 # adjust according to your GPU memory
                pred_rgb = []
                for i in range(0, H*W, chunk_size):
                    rays_o_chunk = rays_o.reshape(-1, 3)[i:i+chunk_size]
                    rays_d_chunk = rays_d.reshape(-1, 3)[i:i+chunk_size]
                    pred_rgb.append(volumn_render(NeRF=nerf, rays_o=rays_o_chunk, rays_d=rays_d_chunk, N_samples=N_samples))
                pred_rgb = torch.cat(pred_rgb, dim=0)
                torchvision.utils.save_image(pred_rgb.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'output/NeRF/pred_{epoch}.png')
                torchvision.utils.save_image(img.reshape(H, W, 3).permute(2, 0, 1).unsqueeze(0), f'output/NeRF/gt_{epoch}.png')

                
if __name__ == '__main__':
    data_path = './data/lego/lego' # data path
    print("Starting NeRF training...")
    psnr_value = fit_images_and_calculate_psnr(data_path)
    print("Training completed.")

    # rays_o, rays_d = sample_rays(100, 100, 100.0, torch.eye(4))
    # print(rays_o[50,50])  # should be (0,0,0) because camera at origin
    # print(rays_d[50,50])  # should be close to (0,0,-1)

    # tn = torch.tensor([[2.0]])
    # tf = torch.tensor([[6.0]])
    # t_vals = sample_points_along_the_ray(tn, tf, 5)
    # print(t_vals)

    # # create encoders
    # xyz_encoder = PositionalEncoder(data_range=[-4, 4], L=10, include_input=False)
    # dir_encoder = PositionalEncoder(data_range=[-1, 1], L=4, include_input=False)

    # # create nerf model
    # nerf = NeRF(
    #     xyz_encoder=xyz_encoder,
    #     dir_encoder=dir_encoder,
    #     input_dim=60,
    #     view_dim=24,
    # )

    # # create dummy rays
    # dummy_rays_o = torch.zeros(10, 3)               # 10 rays at origin
    # dummy_rays_d = torch.tensor([[0, 0, -1.0]]).repeat(10, 1)  # all pointing forward

    # # test volume rendering
    # out = volumn_render(nerf, dummy_rays_o, dummy_rays_d, 64)
    # print("Output shape:", out.shape)


