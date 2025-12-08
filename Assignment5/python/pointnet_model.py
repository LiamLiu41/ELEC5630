# python/pointnet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core Network: PointNet-like Noise Predictor (ε_θ) ---
class PointCloudNoisePredictor(nn.Module):
    def __init__(self, n_points, point_dim, embed_dim, t_steps):
        """
        Args:
            n_points (int): Number of points in the point cloud.
            point_dim (int): Dimension of each point (usually 3 for x,y,z).
            embed_dim (int): Dimension of the feature embeddings.
            t_steps (int): Total number of diffusion steps (used for time embedding scaling if needed).
        """
        super().__init__()
        
        self.n_points = n_points
        self.point_dim = point_dim
        self.t_steps = t_steps
        self.embed_dim = embed_dim
        
        # Time Step Embedding (for conditioning)
        # Projects the scalar time step t into a high-dimensional vector.
        # Input: (B, 1) -> Output: (B, embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # 1. Local Feature Extraction (Per-point MLP)
        # Maps each point (x,y,z) to a higher dimensional feature space independently.
        # Implemented using Conv1d with kernel_size=1 (shared MLP).
        # Input: (B, point_dim, N) -> Output: (B, embed_dim, N)
        self.local_mlp = nn.Sequential(
            nn.Conv1d(point_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU()
        )

        # 2. Global Feature Extraction (Max Pooling)
        # Logic is implemented in the forward pass using torch.max

        # 3. Noise Prediction MLP
        # Combines Local Features + Global Features + Time Embedding to predict noise.
        # Input dimension = Local(embed) + Global(embed) + Time(embed)
        concat_dim = embed_dim * 3
        
        self.output_mlp = nn.Sequential(
            nn.Conv1d(concat_dim, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Conv1d(embed_dim, embed_dim // 2, 1),
            nn.BatchNorm1d(embed_dim // 2),
            nn.GELU(),
            nn.Conv1d(embed_dim // 2, point_dim, 1) # Output dim must match input point_dim (3)
        )

    def forward(self, x, t):
        """
        Forward pass of the PointCloudNoisePredictor.
        
        Args:
            x (torch.Tensor): Input point cloud (B, N_POINTS, POINT_DIM).
            t (torch.Tensor): Time steps (B,).
        
        Returns:
            torch.Tensor: Predicted noise (B, N_POINTS, POINT_DIM).
        """
        
        # x: (B, N, D)
        B, N, D = x.shape
        
        # 1. Transpose input to match Conv1d requirement: (B, N, D) -> (B, D, N)
        x = x.transpose(1, 2)
        
        # 2. Local Feature Extraction
        # local_feat: (B, embed_dim, N)
        local_feat = self.local_mlp(x)
        
        # 3. Global Feature Extraction (Max Pooling)
        # Aggregate features across all points to get a global shape signature.
        # global_feat: (B, embed_dim)
        global_feat = torch.max(local_feat, dim=2)[0]
        
        # 4. Time Embedding Processing
        # Ensure t is float and has the right shape (B, 1) for the Linear layer
        t_input = t.view(-1, 1).float() / self.t_steps # Normalize t slightly
        # t_emb: (B, embed_dim)
        t_emb = self.time_embed(t_input)
        
        # 5. Feature Combination (Concatenation)
        # We need to expand Global and Time features to match the number of points N
        
        # Expand Global: (B, embed_dim) -> (B, embed_dim, N)
        global_feat_expanded = global_feat.unsqueeze(2).repeat(1, 1, N)
        
        # Expand Time: (B, embed_dim) -> (B, embed_dim, N)
        t_emb_expanded = t_emb.unsqueeze(2).repeat(1, 1, N)
        
        # Concatenate along the channel dimension (dim=1)
        # combined: (B, embed_dim * 3, N)
        combined = torch.cat([local_feat, global_feat_expanded, t_emb_expanded], dim=1)
        
        # 6. Predict Noise
        # output: (B, D, N)
        output = self.output_mlp(combined)
        
        # 7. Transpose output back to original format: (B, D, N) -> (B, N, D)
        return output.transpose(1, 2)