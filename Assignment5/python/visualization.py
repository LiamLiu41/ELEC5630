# python/visualization.py

import matplotlib.pyplot as plt

def plot_training_loss(losses, save_path='results/training_loss.png'):
    """
    Plots and saves the training loss curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(losses) + 1), losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average MSE Loss')
    plt.grid(True)
    plt.savefig(save_path)
    print(f"\n[INFO] Training loss plot saved to {save_path}")
    plt.close()

def plot_point_cloud_3d(pc_array, num_samples, save_path='results/generated_samples.png'):
    """
    Visualizes and saves the generated point cloud results in 3D.
    
    Args:
        pc_array (np.ndarray): Point cloud array to visualize (N_samples, N_POINTS, 3).
        num_samples (int): The number of samples to plot.
        save_path (str): Path to save the image.
    """
    # TODO: Implement the visualization logic here, you can use matplotlib/open3d/mistuba for 3D plotting or any other library of your choice.
    # If using mitsuba, please refer https://github.com/stevenygd/PointFlow?tab=readme-ov-file