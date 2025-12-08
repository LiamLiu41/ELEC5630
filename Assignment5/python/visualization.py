# python/visualization.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

def plot_training_loss(losses, save_path='results/training_loss.png'):
    """
    Plots training loss curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', markersize=3, linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved loss plot to {save_path}")

def plot_point_cloud_3d(pc_data, num_samples=4, save_path='results/samples.png'):
    """
    Visualizes point clouds in 3D using Matplotlib.
    
    Args:
        pc_data: Numpy array or Tensor (B, N, 3)
        num_samples: Number of samples to plot
        save_path: Output filename
    """
    # 1. Convert to numpy
    if isinstance(pc_data, torch.Tensor):
        pc_data = pc_data.detach().cpu().numpy()
    
    total_samples = pc_data.shape[0]
    num_samples = min(num_samples, total_samples)
    
    # 2. Setup Plot
    fig = plt.figure(figsize=(4 * num_samples, 4))
    
    for i in range(num_samples):
        ax = fig.add_subplot(1, num_samples, i + 1, projection='3d')
        points = pc_data[i]
        
        # 3. Scatter Plot
        # Swap axes if necessary to make it look upright (z-up)
        # Usually ShapeNet is Y-up, Matplotlib is Z-up.
        ax.scatter(points[:, 0], points[:, 2], points[:, 1], s=2, c='teal', alpha=0.6)
        
        ax.set_title(f"Sample {i}")
        ax.axis('off') # Hide axes for clean look
        
        # Hack to enforce equal aspect ratio in Matplotlib 3D
        try:
            max_range = np.array([points[:,0].max()-points[:,0].min(), 
                                  points[:,1].max()-points[:,1].min(), 
                                  points[:,2].max()-points[:,2].min()]).max() / 2.0
            mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
            mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
            mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        except:
            pass # Skip adjustment if data is weird (e.g. all zeros)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Viz] Saved 3D visualization to {save_path}")