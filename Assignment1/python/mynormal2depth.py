import numpy as np
import cv2
import open3d as o3d
from scipy.fftpack import fft2, ifft2
import os

def mynormal2depth(normal_map, data):
    """
    Convert normal map to depth map using Frankot-Chellappa algorithm.
    
    Parameters:
    - normal_map: Estimated normal map (h, w, 3)
    - data: Data object containing mask and other information
    
    Returns:
    - depth_map: Reconstructed depth map
    - mesh: Triangle mesh generated from point cloud
    """
    
    mask = data.mask
    h, w = mask.shape
    
    print(f"Normal map shape: {normal_map.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # 修复1: 处理法向量中的无效值
    normal_map_clean = np.nan_to_num(normal_map, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 修复2: 确保法向量在掩码外的区域为零
    normal_map_clean[~mask] = [0, 0, 1]  # 设置为默认向上法向量
    
    # Extract surface gradients from normal map
    # For a normal n = (nx, ny, nz), the gradients are:
    # p = -nx/nz, q = -ny/nz
    nz = normal_map_clean[:, :, 2].copy()
    nz[nz == 0] = 1e-8  # 避免除零
    
    p = -normal_map_clean[:, :, 0] / nz
    q = -normal_map_clean[:, :, 1] / nz
    
    # Apply mask
    p[~mask] = 0
    q[~mask] = 0
    
    print(f"Gradient p range: [{np.min(p[mask]):.3f}, {np.max(p[mask]):.3f}]")
    print(f"Gradient q range: [{np.min(q[mask]):.3f}, {np.max(q[mask]):.3f}]")
    
    # Frankot-Chellappa algorithm
    depth_map = frankot_chellappa(p, q, mask)
    
    # Apply mask to depth map
    depth_map[~mask] = 0
    
    # Create point cloud and mesh
    print("Creating mesh from depth map...")
    point_cloud, mesh = create_mesh_from_depth(depth_map, mask, data)
    
    return depth_map, mesh

def frankot_chellappa(p, q, mask):
    """
    Frankot-Chellappa algorithm for surface integration from gradients.
    """
    h, w = p.shape
    
    # Create frequency grids
    u = np.fft.fftfreq(w) * 2 * np.pi
    v = np.fft.fftfreq(h) * 2 * np.pi
    U, V = np.meshgrid(u, v)
    
    # Avoid division by zero at DC component
    denominator = U**2 + V**2
    denominator[0, 0] = 1.0  # Set DC component to avoid division by zero
    
    # Fourier transforms of gradients
    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)
    
    # Compute Fourier transform of depth
    Z = (-1j * U * P - 1j * V * Q) / denominator
    
    # Set DC component to zero (arbitrary base level)
    Z[0, 0] = 0
    
    # Inverse Fourier transform to get depth
    z = np.real(np.fft.ifft2(Z))
    
    # Make depth positive and normalize within mask
    z_masked = z.copy()
    z_masked[~mask] = np.nan
    z_min = np.nanmin(z_masked)
    z = z - z_min
    z[~mask] = 0
    
    print(f"Depth map range: [{np.min(z[mask]):.3f}, {np.max(z[mask]):.3f}]")
    
    return z

def create_mesh_from_depth(depth_map, mask, data):
    """
    Create point cloud and mesh from depth map using Open3D.
    """
    h, w = depth_map.shape
    
    # Create coordinate grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Apply mask - 只处理有效像素
    valid_indices = np.where(mask & (depth_map != 0))
    
    if len(valid_indices[0]) == 0:
        raise ValueError("No valid points found in depth map")
    
    print(f"Creating point cloud with {len(valid_indices[0])} points...")
    
    # Create 3D points
    points = np.zeros((len(valid_indices[0]), 3))
    points[:, 0] = x[valid_indices]  # x coordinate
    points[:, 1] = y[valid_indices]  # y coordinate  
    points[:, 2] = depth_map[valid_indices]  # z coordinate (depth)
    
    # 修复3: 确保没有无穷大或NaN值
    points = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 修复4: 中心化坐标
    points[:, 0] = points[:, 0] - np.mean(points[:, 0])
    points[:, 1] = points[:, 1] - np.mean(points[:, 1])
    points[:, 2] = points[:, 2] - np.mean(points[:, 2])
    
    # 修复5: 检查点云数据范围
    print(f"Point cloud range - X: [{np.min(points[:, 0]):.1f}, {np.max(points[:, 0]):.1f}]")
    print(f"Point cloud range - Y: [{np.min(points[:, 1]):.1f}, {np.max(points[:, 1]):.1f}]")
    print(f"Point cloud range - Z: [{np.min(points[:, 2]):.1f}, {np.max(points[:, 2]):.1f}]")
    
    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # 修复6: 估计法向量（可选，但有助于Poisson重建）
    print("Estimating normals...")
    point_cloud.estimate_normals()
    
    # 修复7: 使用更简单的网格生成方法，避免段错误
    print("Creating mesh...")
    
    # 方法1: 首先尝试泊松重建
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=6  # 降低深度以减少内存使用
        )
        
        # 移除低密度顶点
        if len(densities) > 0:
            density_threshold = np.quantile(densities, 0.01)
            vertices_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
        print(f"Poisson mesh - Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
        
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
        print("Falling back to ball pivoting...")
        
        # 方法2: 如果泊松失败，使用球旋转算法
        try:
            distances = point_cloud.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 2 * avg_dist
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                point_cloud, o3d.utility.DoubleVector([radius, radius * 2])
            )
            print(f"Ball pivoting mesh - Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
            
        except Exception as e2:
            print(f"Ball pivoting also failed: {e2}")
            print("Creating simple point cloud mesh...")
            
            # 方法3: 最后的手段，创建简单的凸包
            mesh, _ = point_cloud.compute_convex_hull()
            print(f"Convex hull mesh - Vertices: {len(mesh.vertices)}, Triangles: {len(mesh.triangles)}")
    
    return point_cloud, mesh

def save_results(depth_map, mesh, data, method_name="robust"):
    """
    Save depth map and mesh to files.
    """
    data_name = data.name if hasattr(data, 'name') else 'unknown'
    results_folder = 'results'
    
    # Create folder for the dataset if it doesn't exist
    data_result_folder = os.path.join(results_folder, f"{data_name}_{method_name}")
    if not os.path.exists(data_result_folder):
        os.makedirs(data_result_folder)
    
    # Save depth map as PNG and NPY
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_path = os.path.join(data_result_folder, f'depth_map_{method_name}.png')
    depth_data_path = os.path.join(data_result_folder, f'depth_map_{method_name}.npy')
    
    cv2.imwrite(depth_map_path, depth_map_normalized)
    np.save(depth_data_path, depth_map)
    
    # Save mesh
    mesh_path = os.path.join(data_result_folder, f'mesh_{method_name}.ply')
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    
    print(f"Depth map saved at {depth_map_path}")
    print(f"Depth data saved at {depth_data_path}")
    print(f"Mesh saved at {mesh_path}")
    
    return depth_map_path, mesh_path