# python/main.py

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt

from pointnet_model import PointCloudNoisePredictor
from evaluation import plot_training_loss, plot_point_cloud_3d # Assuming these functions are available


# --- 0. Configuration Parameters ---
class Config:
    N_POINTS = 2048        # Number of points per point cloud (target sampled size)
    POINT_DIM = 3          # Dimension of points (x, y, z)
    EMBED_DIM = 64         # Feature embedding dimension for PointNet layers
    T_STEPS = 1000         # Total number of diffusion time steps (T)
    
    # TODO: You can modify these training parameters as needed
    BATCH_SIZE = 32        # Batch size for training (can be modified)
    N_EPOCHS = 5           # Training Epochs (can be modified)
    LEARNING_RATE = 1e-4   # Learning Rate (can be modified)
    VAL_STEP = 10          # Perform a validation/checkpoint save every N batches
    
    # Checkpoint file path templates
    CHECKPOINT_PATH_EPOCH = 'results/model_epoch_final.pth' # Path to save model weights after final epoch
    CHECKPOINT_PATH_STEP_TEMPLATE = 'results/model_step_{step:06d}.pth' # Template for intermediate step checkpoints

config = Config()


# --- 1. Data Loading Utility (Includes Random Sampling Logic) ---
def load_shapenet_split(split_name='train', base_dir='../data/03001627'):
    """
    Loads point cloud data for a ShapeNet 03001627 subset and uniformly samples the point count.
    
    Args:
        split_name (str): 'train', 'val', or 'test'.
        base_dir (str): Base directory containing the ShapeNet data split folders.
        
    Returns:
        torch.Tensor: Tensor containing all sampled point clouds (Total_Samples, N_POINTS, 3).
    """
    target_n_points = config.N_POINTS 
    split_path = Path(base_dir) / split_name
    print(f"Attempting to load data from: {split_path}")
    
    if not split_path.is_dir():
        print(f"[ERROR] Directory not found: {split_path}")
        print("Please ensure the '03001627' directory is correctly placed inside '../data/'")
        return None

    point_clouds = []
    files = sorted([f for f in os.listdir(split_path) if f.endswith('.npy')])
    
    if not files:
        print(f"[WARNING] No .npy files found in {split_path}")
        return None

    for file in files:
        path = split_path / file
        try:
            data = np.load(path) # Shape: (N_original, 3)
            N_original = data.shape[0]
            
            # --- Random Sampling/Upsampling Logic to ensure N_POINTS ---
            if N_original != target_n_points:
                if N_original >= target_n_points:
                    # Downsampling (sample without replacement)
                    choice = np.random.choice(N_original, target_n_points, replace=False)
                else:
                    # Upsampling/Padding (sample with replacement)
                    choice = np.random.choice(N_original, target_n_points, replace=True)
                
                data = data[choice, :]

            # TODO: Do we need to normalize or center the point clouds here? If so, add that logic.

            point_clouds.append(torch.tensor(data, dtype=torch.float32))
            
        except Exception as e:
            print(f"[Error] Failed to load {path}: {e}")
            continue
        
    if not point_clouds:
        return None

    stacked_data = torch.stack(point_clouds)
    print(f"[INFO] Data loaded and uniformly sampled/padded to shape: {stacked_data.shape[1:]}")
    
    return stacked_data


# --- 2. Diffusion Scheduler (Forward and Reverse Processes) ---
class DiffusionScheduler:
    def __init__(self, t_steps, device):
        self.T = t_steps
        self.device = device

    # Forward Process Sampling: q(x_t | x_0)
    def forward_sample(self, x0, t):
        """
        Samples x_t given x_0 and time step t using the forward diffusion process.
        
        Args:
            x0 (torch.Tensor): Original point cloud (B, N_POINTS, 3).
            t (torch.Tensor): Time steps (B,).
        Returns:
            x_t (torch.Tensor): Noisy point cloud at time t (B, N_POINTS, 3).
            epsilon (torch.Tensor): The noise added (B, N_POINTS, 3).
        """
        # TODO: Implement the forward diffusion sampling logic here

    # Reverse Process Step: Denoising from x_t to x_{t-1}
    @torch.no_grad()
    def reverse_step(self, xt, t, predicted_noise):
        """
        Performs one reverse diffusion step from x_t to x_{t-1}.
        
        Args:
            xt (torch.Tensor): Noisy point cloud at time t (B, N_POINTS, 3).
            t (int): Current time step.
            predicted_noise (torch.Tensor): Predicted noise by the model (B, N_POINTS, 3).
        
        Returns:
            x_{t-1} (torch.Tensor): Denoised point cloud at time t-1 (B, N_POINTS, 3).
        """
        # TODO: Implement the reverse diffusion step logic here

@torch.no_grad()
def generate(model, scheduler, device, num_samples=4):
    model.eval()
    
    # 1. Start generation from pure noise (x_T)
    xt = torch.randn(num_samples, config.N_POINTS, config.POINT_DIM, device=device)
    
    print(f"\n--- Starting Generation (T={config.T_STEPS} steps) ---")
    
    # TODO: Implement the full generation loop here

def visualize_denoising_process(model, scheduler, device, num_steps=50, save_path='results/denoising_process.png'):
    """
    Visualizes the denoising process of the diffusion model over a specified number of steps.
    
    Args:
        model: The trained diffusion model.
        scheduler: The diffusion scheduler.
        device: The computation device (CPU/GPU).
        num_steps (int): Number of denoising steps to visualize.
        save_path (str): Path to save the visualization image.
    """
    # TODO: Implement the denoising process visualization logic here

# --- 3. Training and Generation Functions ---
def train(model, scheduler, device, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epoch_losses = []
    global_step = 0
    
    print(f"--- Starting Training on {device} ---")
    start_time = time.time()
    
    for epoch in range(config.N_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch_data in enumerate(dataloader):
            batch_data = batch_data[0].to(device)
            B = batch_data.shape[0]
            
            # --- Training Step ---
            # Sample time step t
            t = torch.randint(0, config.T_STEPS, (B,), device=device).long()
            
            # TODO: Complete the training step logic below
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            
            # --- Validation and Checkpoint Step ---
            if global_step % config.VAL_STEP == 0:
                model.eval()
                with torch.no_grad():
                    # Calculate Approximate Validation Loss using the current batch
                    val_loss = F.mse_loss(epsilon, predicted_epsilon).item()
                    
                    print(f"  [VAL Step {global_step}] Batch Loss: {val_loss:.6f}")
                    
                    # Save intermediate checkpoint using the step number
                    try:
                        step_path = config.CHECKPOINT_PATH_STEP_TEMPLATE.format(step=global_step)
                        torch.save(model.state_dict(), step_path)
                        print(f"  [CHECKPOINT] Model weights saved to {step_path}")
                    except Exception as e:
                        print(f"  [ERROR] Failed to save step checkpoint: {e}")
                model.train() # Return to training mode
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{config.N_EPOCHS}] Average Loss: {avg_loss:.6f}")
        epoch_losses.append(avg_loss)

    end_time = time.time()
    print(f"Training finished in {end_time - start_time:.2f} seconds.")
    
    # --- SAVE FINAL MODEL WEIGHTS ---
    try:
        # Save the final model's state dictionary
        torch.save(model.state_dict(), config.CHECKPOINT_PATH_EPOCH)
        print(f"[INFO] Final model weights saved to {config.CHECKPOINT_PATH_EPOCH}")
    except Exception as e:
        print(f"[ERROR] Failed to save final model weights: {e}")
    
    # --- Training Visualization (imported from evaluation.py) ---
    plot_training_loss(epoch_losses)


# --- 4. Main Program Entry Point ---
def main():
    # Ensure output directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('python', exist_ok=True) 

    # Determine the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load data (includes sampling logic)
    all_data = load_shapenet_split('train')
    
    if all_data is None:
        print("[FATAL] Could not load training data. Exiting.")
        return
    
    print(f"Loaded {all_data.shape[0]} training samples with unified shape {all_data.shape[1:]}")
    
    # 2. Convert to PyTorch DataLoader
    dataset = TensorDataset(all_data)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 3. Initialize Model and Scheduler
    model = PointCloudNoisePredictor(
        config.N_POINTS, 
        config.POINT_DIM, 
        config.EMBED_DIM, 
        config.T_STEPS
    ).to(device)
    
    scheduler = DiffusionScheduler(config.T_STEPS, device)

    # 4. Train model and save checkpoints
    train(model, scheduler, device, dataloader)

    # 5. Generate samples using the trained model
    generated_samples = generate(model, scheduler, device, num_samples=4)
    visualize_denoising_process(model, scheduler, device, num_steps=50, save_path='results/denoising_process.png')

    # 6. Save final generated point clouds
    np.save('results/generated_points.npy', generated_samples)
    print("\n[INFO] Generated samples saved to results/generated_points.npy")


if __name__ == '__main__':
    main()