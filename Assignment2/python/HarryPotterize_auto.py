# # Q2.6

# import cv2
# import numpy as np

# from .matchPics import matchPics
# from .computeH_ransac import computeH_ransac
# from .warpH import warpH
# from .compositeH import compositeH

# # Load images
# cv_img = cv2.imread('../data/cv_cover.jpg')
# desk_img = cv2.imread('../data/cv_desk.png')
# hp_img = cv2.imread('../data/hp_cover.jpg')

# # Extract features and match
# locs1, locs2 = matchPics(cv_img, desk_img)

# # Compute homography using RANSAC
# bestH2to1, _ = computeH_ransac(locs1, locs2);

# # Scale harry potter image to template size
# scaled_hp_img = cv2.resize(hp_img, (desk_img.shape[1], desk_img.shape[0]))

# # Display warped image
# warped_hp_img = warpH(scaled_hp_img, bestH2to1, desk_img.shape)
# cv2.imshow('Warped Image', warped_hp_img)
# cv2.waitKey(0)

# # Display composite image
# composite_img = compositeH(bestH2to1, scaled_hp_img, desk_img)
# cv2.imshow('Composite Image', composite_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Q2.6

# Q2.6

import cv2
import numpy as np

from matchPics import matchPics
from computeH_ransac import computeH_ransac
from warpH import warpH
from compositeH import compositeH

# Load images
cv_img = cv2.imread('../data/cv_cover.jpg')
desk_img = cv2.imread('../data/cv_desk.png')
hp_img = cv2.imread('../data/hp_cover.jpg')

# Extract features and match
keypoints1, keypoints2 = matchPics(cv_img, desk_img)

# Extract coordinates from keypoints
locs1 = np.array([kp.pt for kp in keypoints1], dtype=np.float32)
locs2 = np.array([kp.pt for kp in keypoints2], dtype=np.float32)

# Print the shapes of locs1 and locs2 for debugging
print(f"Number of matched keypoints in image 1: {locs1.shape[0]}")
print(f"Number of matched keypoints in image 2: {locs2.shape[0]}")

# Check if locs1 and locs2 have the same number of matches
if locs1.shape[0] == 0 or locs2.shape[0] == 0:
    print("No keypoints matched. Check the input images and feature extraction.")
else:
    # Use the minimum number of matches
    min_matches = min(locs1.shape[0], locs2.shape[0])
    locs1 = locs1[:min_matches]
    locs2 = locs2[:min_matches]

    # Compute homography using RANSAC
    bestH2to1, inliers = computeH_ransac(locs1, locs2)

    # Scale harry potter image to template size
    scaled_hp_img = cv2.resize(hp_img, (desk_img.shape[1], desk_img.shape[0]))

    # Display and save warped image
    warped_hp_img = warpH(scaled_hp_img, bestH2to1, desk_img.shape)
    cv2.imwrite('../results/warped_hp_image.png', warped_hp_img)

    # Display and save composite image
    composite_img = compositeH(bestH2to1, scaled_hp_img, desk_img)
    cv2.imwrite('../results/composite_image.png', composite_img)

    # Optionally visualize inlier matches
    if inliers.any():
        inlier_locs1 = locs1[inliers]
        inlier_locs2 = locs2[inliers]
        for p1, p2 in zip(inlier_locs1, inlier_locs2):
            cv2.circle(cv_img, tuple(p1.astype(int)), 5, (0, 255, 0), -1)  # Green circles on image 1
            cv2.circle(desk_img, tuple(p2.astype(int)), 5, (0, 255, 0), -1)  # Green circles on image 2

        # Save inlier matches images
        cv2.imwrite('../results/inlier_matches_image1.png', cv_img)
        cv2.imwrite('../results/inlier_matches_image2.png', desk_img)

# Clean up
cv2.destroyAllWindows()