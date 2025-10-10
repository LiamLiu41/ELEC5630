import numpy as np
import cv2
import os
from datetime import datetime
from sklearn.decomposition import PCA

def myPCA(data, resize):
    """
    PCA-based Photometric Stereo algorithm for estimating normal maps.
    
    Parameters:
    - data: An object containing:
        - 'imgs': List of images (numpy arrays).
        - 'mask': A binary mask image (numpy array).
        - 'L': A (3, N) numpy array of light directions.
    - resize: Optional. A tuple (width, height) to resize images.
    
    Returns:
    - normal_map: The estimated normal map (numpy array).
    """
    
    images = data.imgs.reshape((-1, resize[0], resize[1]))  
    mask = data.mask
    light_directions = data.L

    print(f"Images shape: {images.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Light directions shape: {light_directions.shape}")

    h, w = images[0].shape
    num_images = len(images)

    # Only keep intensity values where mask is non-zero
    valid_pixel_indices = np.where(mask.flatten() > 0)[0]
    print(f"Number of valid pixels: {len(valid_pixel_indices)}")

    # 创建强度矩阵: (num_valid_pixels, num_images)
    I = np.array([img.flatten()[valid_pixel_indices] for img in images]).T
    L = light_directions.T  # (num_images, 3)

    print(f"I shape: {I.shape}")
    print(f"L shape: {L.shape}")

    # Check if dimensions match
    if I.shape[1] != L.shape[0]:
        raise ValueError(f"Number of images does not match the number of light directions: {I.shape[1]} vs {L.shape[0]}")

    # 方法1: 直接使用伪逆求解 (推荐)
    print("Using direct pseudo-inverse method...")
    try:
        # 直接求解: L @ N = I^T
        # 所以 N = pinv(L) @ I^T
        L_pinv = np.linalg.pinv(L)  # (3, num_images)
        normal_map_vec = L_pinv @ I.T  # (3, num_valid_pixels)
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError in direct method: {e}")
        # 方法2: 使用最小二乘求解
        print("Falling back to least squares method...")
        normal_map_vec = np.linalg.lstsq(L, I.T, rcond=None)[0]  # (3, num_valid_pixels)

    print(f"normal_map_vec shape: {normal_map_vec.shape}")

    # Initialize normal_map with zeros
    normal_map = np.zeros((h, w, 3), dtype=np.float32)

    # Assign the values from normal_map_vec to the corresponding valid pixels
    normal_map_flat = normal_map.reshape((-1, 3))
    normal_map_flat[valid_pixel_indices] = normal_map_vec.T  # 转置为 (num_valid_pixels, 3)

    # Normalize the normal map
    norm = np.linalg.norm(normal_map_flat, axis=1, keepdims=True)
    normal_map_flat /= np.where(norm == 0, 1, norm)  # 避免除以零

    # 保存结果
    data_name = data.name if hasattr(data, 'name') else 'unknown'
    results_folder = 'results'
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_result_folder = os.path.join(results_folder, f"{data_name}_pca")
    if not os.path.exists(data_result_folder):
        os.makedirs(data_result_folder)

    normal_map_path = os.path.join(data_result_folder, 'normal_map_pca.png')
    normal_data_path = os.path.join(data_result_folder, 'normal_map_pca.npy')

    normal_map_normalized = ((normal_map + 1) / 2 * 255).astype(np.uint8)
    cv2.imwrite(normal_map_path, normal_map_normalized)
    np.save(normal_data_path, normal_map)

    print(f"Normal map saved at {normal_map_path}")
    print(f"Normal data saved at {normal_data_path}")

    return normal_map