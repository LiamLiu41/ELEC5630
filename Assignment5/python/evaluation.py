# python/evaluation_metrics.py

import torch
import numpy as np

def compute_nearest_neighbor_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared distance from each point in pc1 to its nearest neighbor in pc2.
    
    Args:
        pc1 (torch.Tensor): Point cloud 1 (B, N1, D).
        pc2 (torch.Tensor): Point cloud 2 (B, N2, D).
        
    Returns:
        torch.Tensor: Squared distance from pc1 to pc2 (B, N1).
    """
    B, N1, D = pc1.shape
    N2 = pc2.shape[1]
    
    # Expand tensors to facilitate distance calculation across all pairs
    # pc1_expanded: (B, N1, 1, D)
    # pc2_expanded: (B, 1, N2, D)
    pc1_expanded = pc1.unsqueeze(2)
    pc2_expanded = pc2.unsqueeze(1)
    
    # Calculate squared Euclidean distance: ||pc1_i - pc2_j||^2
    # The result 'dist_sq': (B, N1, N2)
    # (B, N1, N2, D) -> (B, N1, N2) by summing over dimension D
    dist_sq = torch.sum((pc1_expanded - pc2_expanded) ** 2, dim=-1)
    
    # Find the minimum squared distance for each point in pc1 across pc2
    # min_dist_sq: (B, N1)
    min_dist_sq, _ = torch.min(dist_sq, dim=2)
    
    return min_dist_sq


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor, reduce: bool = True) -> torch.Tensor:
    """
    Computes the Chamfer Distance between two batched point clouds.
    
    CD(P1, P2) = sum_{x in P1} min_{y in P2} ||x - y||^2 + sum_{y in P2} min_{x in P1} ||y - x||^2
    
    Args:
        pc1 (torch.Tensor): Point cloud 1 (B, N, D).
        pc2 (torch.Tensor): Point cloud 2 (B, N, D).
        reduce (bool): If True, returns the mean CD across the batch.
                       If False, returns the CD for each batch element (B,).
        
    Returns:
        torch.Tensor: The Chamfer Distance.
    """
    # TODO: Implement the Chamfer Distance calculation using the nearest neighbor distances


def minimum_matching_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Minimum Matching Distance (MMD) between two sets of point clouds.
    
    MMD(P1, P2) = (1/|P1|) * sum_{p1 in P1} min_{p2 in P2} CD(p1, p2)
    
    Args:
        pc1 (torch.Tensor): Set of point clouds 1 (B1, N, D).
        pc2 (torch.Tensor): Set of point clouds 2 (B2, N, D).
        
    Returns:
        torch.Tensor: The Minimum Matching Distance.
    """
    # TODO: Implement the Minimum Matching Distance calculation using Chamfer Distance


# --- Example Usage (Optional: For testing the metric) ---
if __name__ == '__main__':
    # 1. Load data
    generated_pc_np = np.load(generated_path)
    generated_pc = torch.tensor(generated_pc_np, dtype=torch.float32).to(device)
    gt_pc = gt_pc.to(device)
    # 2. Load ground truth point cloud
    from .point_cloud_diffusion import load_shapenet_split
    gt_pc = load_shapenet_split("train")
    print(f"Loaded Generated PC shape: {generated_pc_np.shape}")
    print(f"Loaded Ground Truth PC shape: {gt_pc.shape}")

    # 3. Compute minimum matching distance
    mmd = minimum_matching_distance(generated_pc, gt_pc)
    print(f"Minimum Matching Distance (MMD): {mmd.item()}")
