# import cv2
# import numpy as np

# def compositeH(H2to1, template, img):
#     # Note that the homography we compute is from the image to the template;
#     # x_template = H2to1 * x_photo
#     # For warping the template to the image, we need to invert it.
#     H_template_to_img = np.linalg.inv(H2to1)

#     # Create mask of same size as template


#     # Warp mask by appropriate homography


#     # Warp template by appropriate homography


#     # Use mask to combine the warped template and the image


#     pass


import cv2
import numpy as np
from warpH import warpH

def compositeH(H2to1, template, img):
    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1 * x_photo
    # For warping the template to the image, we need to invert it.
    H_template_to_img = np.linalg.inv(H2to1)

    # Create mask of same size as template
    mask = np.zeros((template.shape[0], template.shape[1]), dtype=np.uint8)
    mask[:] = 255  # White mask

    # Warp mask by appropriate homography
    warped_mask = warpH(mask, H_template_to_img, img.shape[:2])

    # Warp template by appropriate homography
    warped_template = warpH(template, H_template_to_img, img.shape[:2])

    # Use mask to combine the warped template and the image
    composite = img.copy()
    for c in range(3):  # For each channel
        composite[:, :, c] = warped_template[:, :, c] * (warped_mask / 255) + composite[:, :, c] * (1 - warped_mask / 255)

    return composite