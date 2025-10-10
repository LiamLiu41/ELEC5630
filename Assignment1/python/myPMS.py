import os
import numpy as np
import cv2
from datetime import datetime

def myPMS(data, resize):
    """
    Least Squares-Based Photometric Stereo algorithm.

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

    if resize:
        images = [cv2.resize(img, resize) for img in images]

    h, w = images[0].shape
    num_images = len(images)

    # Only keep intensity values where mask is non-zero
    valid_pixel_indices = np.where(mask.flatten() > 0)[0]
    print(f"Number of valid pixels: {len(valid_pixel_indices)}")

    I = np.array([img.flatten()[valid_pixel_indices] for img in images]).T  # Shape (num_valid_pixels, num_images)
    L = light_directions.T  # Shape (num_images, 3)

    print(f"I shape: {I.shape}")
    print(f"L shape: {L.shape}")

    # Check if dimensions match
    if I.shape[1] != L.shape[0]:
        raise ValueError(f"Number of images does not match the number of light directions: {I.shape[1]} vs {L.shape[0]}")

    try:
        # Transpose I to match the expected dimensions
        I = I.T  # Shape (num_images, num_valid_pixels)

        # lstsq expects L to have shape (num_images, 3) and I to have shape (num_images, num_valid_pixels)
        N = np.linalg.lstsq(L, I, rcond=None)[0]  # Shape (3, num_valid_pixels)
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e}")
        print(f"Check shapes: I shape: {I.shape}, L shape: {L.shape}")
        raise

    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    normal_map[mask > 0] = N.T  # Shape (3, num_valid_pixels) to (h, w, 3)

    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map /= norm  # Normalize the normal map

    # Get dataset name from the data object (assumed to have 'name' attribute)
    data_name = data.name if hasattr(data, 'name') else 'unknown'

    # Define results folder path
    results_folder = 'results'

    # Add timestamp to the folder name to avoid overwriting
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create folder for the dataset if it doesn't exist
    data_result_folder = os.path.join(results_folder, f"{data_name}")
    if not os.path.exists(data_result_folder):
        os.makedirs(data_result_folder)

    # Generate paths for saving normal map as PNG and as NumPy array
    normal_map_path = os.path.join(data_result_folder, 'normal_map.png')
    normal_data_path = os.path.join(data_result_folder, 'normal_map.npy')

    # Save the normal map as PNG (normalized to [0, 255])
    normal_map_normalized = ((normal_map + 1) / 2 * 255).astype(np.uint8)
    cv2.imwrite(normal_map_path, normal_map_normalized)

    # Save the normal map as a NumPy file
    np.save(normal_data_path, normal_map)

    print(f"Normal map saved at {normal_map_path}")
    print(f"Normal data saved at {normal_data_path}")

    return normal_map
