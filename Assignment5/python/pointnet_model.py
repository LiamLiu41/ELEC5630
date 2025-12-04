# python/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core Network: PointNet-like Noise Predictor (ε_θ) ---
class PointCloudNoisePredictor(nn.Module):
    def __init__(self, n_points, point_dim, embed_dim, t_steps):
        super().__init__()
        
        self.n_points = n_points
        self.point_dim = point_dim
        self.t_steps = t_steps
        
        # Time Step Embedding (for conditioning)
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # 1. Local Feature Extraction (Per-point MLP)
        # TODO: Define the MLP layers for local feature extraction

        # 2. Global Feature Extraction (Max Pooling)
        # TODO: Define the global feature extraction logic

        # 3. Noise Prediction MLP
        # TODO: Define the MLP layers for noise prediction

    def forward(self, x, t):
        """
        Forward pass of the PointCloudNoisePredictor.
        
        Args:
            x (torch.Tensor): Input point cloud (B, N_POINTS, POINT_DIM).
            t (torch.Tensor): Time steps (B,).
        
        Returns:
            torch.Tensor: Predicted noise (B, N_POINTS, POINT_DIM).
        """
        
        # x: (B, N_POINTS, POINT_DIM)
        B, N, D = x.shape
        
        # TODO: Implement the forward pass logic here