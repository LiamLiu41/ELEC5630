# python/point_cloud_diffusion.py

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import os
from pathlib import Path
from tqdm import tqdm # 如果没有安装 tqdm，可以注释掉相关行

# --- Imports from your custom modules ---
from pointnet_model import PointCloudNoisePredictor
from visualization import plot_training_loss, plot_point_cloud_3d
# from evaluation import chamfer_distance # 可选：如果在训练中需要评估指标

# --- 0. Configuration Parameters ---
class Config:
    # Model Hyperparameters
    N_POINTS = 2048        # Number of points per point cloud
    POINT_DIM = 3          # x, y, z
    EMBED_DIM = 64         # PointNet embedding dimension
    T_STEPS = 1000         # Total diffusion steps
    
    # Training Hyperparameters
    BATCH_SIZE = 32
    N_EPOCHS = 50          # 建议至少跑 50-100 epochs 才能看到像样的结果
    LEARNING_RATE = 1e-4
    VAL_STEP = 500         # 每多少个 Batch 保存一次检查点
    
    # Paths
    CHECKPOINT_PATH_EPOCH = 'results/model_epoch_final.pth'
    CHECKPOINT_PATH_STEP_TEMPLATE = 'results/model_step_{step:06d}.pth'
    DATA_DIR = '../data/03001627' # 请根据你实际的数据路径修改

config = Config()

# --- Helper Function for Broadcasting ---
def extract(a, t, x_shape):
    """
    Extract coefficients at specified time steps t and reshape for broadcasting.
    Args:
        a: (T,) Tensor (schedule parameter)
        t: (B,) Tensor of time indices
        x_shape: Tuple (B, N, D)
    Returns:
        (B, 1, 1) Tensor
    """
    batch_size = t.shape[0]
    
    # 修正：确保索引 t 和参数 a 在同一个设备上
    out = a.gather(-1, t.to(a.device))
    
    # Reshape 并确保最终输出回到 t 所在的设备（通常也是计算设备）
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# --- 1. Data Loading Utility ---
def load_shapenet_split(split_name='train', base_dir=config.DATA_DIR):
    """
    Loads ShapeNet .npy point clouds.
    """
    target_n_points = config.N_POINTS 
    split_path = Path(base_dir) / split_name
    print(f"[INFO] Loading data from: {split_path}")
    
    if not split_path.is_dir():
        print(f"[ERROR] Directory not found: {split_path}")
        # Return dummy data for debugging if real data is missing
        print("[WARNING] Generating DUMMY data for testing code flow...")
        return torch.randn(100, target_n_points, 3)

    point_clouds = []
    files = sorted([f for f in os.listdir(split_path) if f.endswith('.npy')])
    
    if not files:
        print(f"[WARNING] No .npy files found in {split_path}")
        return None

    for file in files:
        path = split_path / file
        try:
            data = np.load(path) # (N_original, 3)
            N_original = data.shape[0]
            
            # Sampling logic to ensure fixed N_POINTS
            if N_original != target_n_points:
                if N_original >= target_n_points:
                    choice = np.random.choice(N_original, target_n_points, replace=False)
                else:
                    choice = np.random.choice(N_original, target_n_points, replace=True)
                data = data[choice, :]

            # Normalization (Center to unit sphere is usually good practice)
            data = data - np.mean(data, axis=0)
            max_dist = np.max(np.sqrt(np.sum(data**2, axis=1)))
            if max_dist > 0:
                data = data / max_dist

            point_clouds.append(torch.tensor(data, dtype=torch.float32))
        except Exception as e:
            continue
        
    if not point_clouds:
        return None

    return torch.stack(point_clouds)

# --- 2. Diffusion Scheduler ---
class DiffusionScheduler:
    def __init__(self, t_steps, device, beta_start=0.0001, beta_end=0.02):
        self.T = t_steps
        self.device = device
        
        # 1. Linear Beta Schedule
        self.betas = torch.linspace(beta_start, beta_end, t_steps).to(device)
        
        # 2. Alpha calculations
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(device)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 3. Forward Process Coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # 4. Posterior Variance (for Reverse Process)
        # sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        # usually simplified to fixed sigma_t^2 = beta_t in simple DDPM
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward_sample(self, x0, t):
        """
        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)I)
        """
        noise = torch.randn_like(x0)
        
        sqrt_alpha_bar_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise

    @torch.no_grad()
    def reverse_step(self, xt, t, predicted_noise):
        """
        p_theta(x_{t-1} | x_t)
        """
        beta_t = extract(self.betas, t, xt.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        sqrt_recip_alpha_t = extract(torch.sqrt(1.0 / self.alphas), t, xt.shape)
        
        # Compute mean
        model_mean = sqrt_recip_alpha_t * (xt - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
        
        # Add noise if t > 0
        if t[0] > 0:
            noise = torch.randn_like(xt)
            # Simple variance schedule: sigma_t = sqrt(beta_t)
            sigma_t = torch.sqrt(beta_t) 
            xt_prev = model_mean + sigma_t * noise
        else:
            xt_prev = model_mean
            
        return xt_prev

# --- 3. Generation Function ---
@torch.no_grad()
def generate(model, scheduler, device, num_samples=4):
    model.eval()
    xt = torch.randn(num_samples, config.N_POINTS, config.POINT_DIM, device=device)
    
    print(f"\n[INFO] Generating {num_samples} samples...")
    iterator = tqdm(reversed(range(0, config.T_STEPS)), total=config.T_STEPS, desc="Sampling")
    
    for i in iterator:
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        predicted_noise = model(xt, t)
        xt = scheduler.reverse_step(xt, t, predicted_noise)
        
    return xt

def visualize_denoising_process(model, scheduler, device, save_path='results/denoising_process.png'):
    """
    Captures a few snapshots during the reverse process for visualization.
    """
    model.eval()
    xt = torch.randn(1, config.N_POINTS, config.POINT_DIM, device=device)
    snapshots = []
    capture_points = [999, 800, 500, 200, 0] # Steps to capture
    
    with torch.no_grad():
        for i in reversed(range(0, config.T_STEPS)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            pred = model(xt, t)
            xt = scheduler.reverse_step(xt, t, pred)
            
            if i in capture_points:
                snapshots.append(xt.cpu().numpy()[0])
    
    # Save simple visualization
    # Ideally reuse plot_point_cloud_3d but modify for multiple steps. 
    # For now, we just save the final result to ensure code runs.
    plot_point_cloud_3d(np.array(snapshots), num_samples=len(snapshots), save_path=save_path)


# --- 4. Training Loop ---
def train(model, scheduler, device, dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    epoch_losses = []
    global_step = 0
    
    print(f"--- Starting Training on {device} ---")
    start_time = time.time()
    
    for epoch in range(config.N_EPOCHS):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.N_EPOCHS}")
        for batch_data in pbar:
            batch_data = batch_data[0].to(device) # (B, N, 3)
            B = batch_data.shape[0]
            
            # 1. Sample t
            t = torch.randint(0, config.T_STEPS, (B,), device=device).long()
            
            # 2. Forward Diffusion
            x_t, epsilon = scheduler.forward_sample(batch_data, t)
            
            # 3. Predict Noise
            predicted_epsilon = model(x_t, t)
            
            # 4. Loss
            loss = F.mse_loss(predicted_epsilon, epsilon)
            
            # 5. Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': loss.item()})
            
            # Optional: Save intermediate checkpoint
            if global_step % config.VAL_STEP == 0:
                pass # Checkpoint logic here if needed
            
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    # Save final model
    torch.save(model.state_dict(), config.CHECKPOINT_PATH_EPOCH)
    print(f"[INFO] Model saved to {config.CHECKPOINT_PATH_EPOCH}")
    
    # Plot loss
    plot_training_loss(epoch_losses)


# --- 5. Main ---
def main():
    os.makedirs('results', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    all_data = load_shapenet_split('train')
    if all_data is None:
        print("[ERROR] Failed to load data. Exiting.")
        return

    dataset = TensorDataset(all_data)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Init Model & Scheduler
    model = PointCloudNoisePredictor(
        config.N_POINTS, config.POINT_DIM, config.EMBED_DIM, config.T_STEPS
    ).to(device)
    
    scheduler = DiffusionScheduler(config.T_STEPS, device)

    # 3. Train
    train(model, scheduler, device, dataloader)

    # 4. Generate & Visualize
    generated_data = generate(model, scheduler, device, num_samples=4)
    
    # Save results
    np.save('results/generated_points.npy', generated_data.cpu().numpy())
    plot_point_cloud_3d(generated_data, num_samples=4, save_path='results/final_samples.png')
    
    # Visualizing the process (Optional)
    visualize_denoising_process(model, scheduler, device)

if __name__ == '__main__':
    main()