# python/evaluation.py

import torch
import torch.nn.functional as F

def compute_pairwise_distance(pc1, pc2):
    """
    Computes the squared Euclidean distance between every pair of points.
    
    Args:
        pc1: (B, N, D)
        pc2: (B, M, D)
    Returns:
        dist: (B, N, M) where dist[b, i, j] = ||pc1[b, i] - pc2[b, j]||^2
    """
    # Expanded to (B, N, 1, D) and (B, 1, M, D)
    diff = pc1.unsqueeze(2) - pc2.unsqueeze(1)
    # Sum over the feature dimension D -> (B, N, M)
    dist = torch.sum(diff ** 2, dim=-1)
    return dist

def chamfer_distance(pc1, pc2):
    """
    Computes the Chamfer Distance between batches of point clouds.
    Formula: Sum_{x in X} min_{y in Y} ||x-y||^2 + Sum_{y in Y} min_{x in X} ||y-x||^2
    
    Args:
        pc1 (torch.Tensor): (B, N, D)
        pc2 (torch.Tensor): (B, M, D)
        
    Returns:
        cd (torch.Tensor): (B,) Chamfer distance for each cloud in the batch.
    """
    # 1. Compute pairwise distance matrix (B, N, M)
    dist = compute_pairwise_distance(pc1, pc2)
    
    # 2. For each point in pc1, find min dist to pc2 (min over dim 2)
    # min_dist1: (B, N)
    min_dist1, _ = torch.min(dist, dim=2)
    
    # 3. For each point in pc2, find min dist to pc1 (min over dim 1)
    # min_dist2: (B, M)
    min_dist2, _ = torch.min(dist, dim=1)
    
    # 4. Sum up the distances (as per assignment formula)
    term1 = torch.sum(min_dist1, dim=1) # (B,)
    term2 = torch.sum(min_dist2, dim=1) # (B,)
    
    return term1 + term2

def minimum_matching_distance(generated_set, reference_set, batch_size=16):
    """
    Computes MMD between a generated set and a reference (ground truth) set.
    MMD = Mean_{Y in Ref} [ Min_{X in Gen} CD(X, Y) ]
    
    Args:
        generated_set (torch.Tensor): (Size_G, N, D)
        reference_set (torch.Tensor): (Size_R, N, D)
        batch_size (int): Batch size for computing CD to avoid OOM.
        
    Returns:
        mmd (torch.Tensor): Scalar MMD value.
    """
    Size_G = generated_set.shape[0]
    Size_R = reference_set.shape[0]
    
    min_cds = []
    
    device = generated_set.device
    
    # Iterate over reference set (Ground Truth)
    # We want to find how well each GT sample is covered by the generated set.
    print(f"[Evaluation] Computing MMD between {Size_G} gen and {Size_R} ref samples...")
    
    # To speed up, we process reference samples in batches
    for i in range(0, Size_R, batch_size):
        end = min(i + batch_size, Size_R)
        ref_batch = reference_set[i:end] # (B_ref, N, D)
        curr_batch_size = ref_batch.shape[0]
        
        # For this batch of reference, we need to find the BEST match in ALL generated samples.
        # This is expensive: O(Size_R * Size_G). 
        # We can loop over Generated set in chunks as well.
        
        best_cds_for_batch = torch.full((curr_batch_size,), float('inf'), device=device)
        
        for j in range(0, Size_G, batch_size):
            gend = min(j + batch_size, Size_G)
            gen_batch = generated_set[j:gend] # (B_gen, N, D)
            
            # Since chamfer_distance expects same batch size (B, N, D) vs (B, N, D),
            # we need to broadcast manually or loop. 
            # Given PyTorch constraints, it's easier to expand dimensions.
            
            # Let's compute CD matrix between ref_batch and gen_batch
            # We want pairwise CD matrix (B_ref, B_gen).
            # This implementation below is a bit complex, let's simplify for readability:
            # We will just loop single reference sample against batch of generated for safety.
            pass

    # Simplified Implementation (Slower but safer memory-wise)
    # Iterate every single Reference Sample
    for i in range(Size_R):
        ref_sample = reference_set[i].unsqueeze(0) # (1, N, D)
        
        # Expand ref to match generated batch size to compute CD in parallel
        # But generated set might be huge, so we process generated set in batches
        
        current_ref_min_cd = float('inf')
        
        for j in range(0, Size_G, batch_size):
            gend = min(j + batch_size, Size_G)
            gen_batch = generated_set[j:gend] # (B_curr, N, D)
            B_curr = gen_batch.shape[0]
            
            ref_expanded = ref_sample.repeat(B_curr, 1, 1) # (B_curr, N, D)
            
            # Compute CD for this chunk
            cds = chamfer_distance(gen_batch, ref_expanded) # (B_curr,)
            
            min_in_chunk = torch.min(cds).item()
            if min_in_chunk < current_ref_min_cd:
                current_ref_min_cd = min_in_chunk
        
        min_cds.append(current_ref_min_cd)
        
    mmd = torch.tensor(min_cds).mean()
    return mmd