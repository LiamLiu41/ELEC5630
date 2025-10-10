import os
import numpy as np
import cv2
from datetime import datetime

def myRobustPMS(data, resize, discard_percentage=10):
    """
    Robust Photometric Stereo algorithm that deals with shadows and highlights.
    
    Parameters:
    - data: An object containing:
        - 'imgs': List of images (numpy arrays).
        - 'mask': A binary mask image (numpy array).
        - 'L': A (3, N) numpy array of light directions.
    - resize: Optional. A tuple (width, height) to resize images.
    - discard_percentage: The percentage of darkest and brightest pixels to discard.
    
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

    # Handle shadows and highlights by discarding the darkest and brightest pixels
    num_valid_pixels = I.shape[0]
    num_discard = int(num_valid_pixels * discard_percentage / 100)

    # Ensure there's enough data to discard (especially if discard_percentage is small)
    if num_discard == 0:
        I_filtered = I  # No filtering if discard_percentage is 0
    else:
        print("==== num_discard is not 0")
        # For each pixel, sort intensities and discard darkest and brightest values
        I_filtered = np.zeros_like(I)
        for i in range(num_valid_pixels):
            pixel_values = I[i, :]  # Intensities for this pixel across all images
            sorted_indices = np.argsort(pixel_values)  # Sort indices based on intensity
            
            # Avoid discarding too much data
            if len(sorted_indices[num_discard:-num_discard]) > 0:
                I_filtered[i, :] = pixel_values[sorted_indices[num_discard:-num_discard]]
            else:
                # If not enough data remains after discarding, keep the original values
                I_filtered[i, :] = pixel_values

    # Ensure dimensions are compatible for least squares
    I_filtered_transposed = I_filtered.T  # Shape (num_images, num_valid_pixels - discarded)

    # Perform least squares to estimate normals
    try:
        N = np.linalg.lstsq(L, I_filtered_transposed, rcond=None)[0]  # Solve for normals
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError: {e}")
        print(f"Check shapes: I_filtered_transposed shape: {I_filtered_transposed.shape}, L shape: {L.shape}")
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
    data_result_folder = os.path.join(results_folder, f"{data_name}_robust")
    if not os.path.exists(data_result_folder):
        os.makedirs(data_result_folder)

    # Generate paths for saving normal map as PNG and as NumPy array
    normal_map_path = os.path.join(data_result_folder, 'normal_map_robust.png')
    normal_data_path = os.path.join(data_result_folder, 'normal_map_robust.npy')

    # Save the normal map as PNG (normalized to [0, 255])
    normal_map_normalized = ((normal_map + 1) / 2 * 255).astype(np.uint8)
    cv2.imwrite(normal_map_path, normal_map_normalized)

    # Save the normal map as a NumPy file
    np.save(normal_data_path, normal_map)

    print(f"Normal map saved at {normal_map_path}")
    print(f"Normal data saved at {normal_data_path}")

    return normal_map
