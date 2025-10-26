# import cv2
# import numpy as np

# # Read the image and convert to grayscale if necessary

# # Create a histogram to store the results


# # Compute the features and descriptors for the original image


# for i in range(37):
#     pass
#     # Rotate the image


#     # Compute features and descriptors for the rotated image

#     # Match features


#     # Apply ratio test


#     # Update histogram


# # Display histogram


# Q2.2 BRIEF and Rotation
# Q2.2 BRIEF and Rotation (Server-safe version)

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # ✅ 防止 Matplotlib 尝试打开窗口
import matplotlib.pyplot as plt
from matchPics import matchPics, matches
import os

# Create output folder
os.makedirs('../results', exist_ok=True)

# Read the image and convert to grayscale if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
if cv_cover is None:
    raise FileNotFoundError("Cannot find '../data/cv_cover.jpg'. Please check the path.")

# Create a histogram to store the results
num_matches = []
angles = list(range(0, 370, 10))  # include 360°

def rotateImage(image, angle):
    """Rotate image around its center without cropping."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (nW, nH))
    return rotated


for i in range(37):  # 0°~360° every 10°
    angle = i * 10
    rotated_img = rotateImage(cv_cover, angle)

    # Match features
    locs1, locs2 = matchPics(cv_cover, rotated_img)
    from matchPics import matches
    count = len(matches)
    num_matches.append(count)
    print(f"Rotation {angle:3d}° -> {count:4d} matches")

    # Save visualized match results for a few rotations
    if angle in [0, 90, 180]:
        match_vis = cv2.drawMatches(cv_cover, locs1, rotated_img, locs2, matches[:40], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        save_path = f"../results/match_rotation_{angle}.png"
        cv2.imwrite(save_path, match_vis)
        print(f"Saved match visualization at {angle}° to {save_path}")

# Save histogram (no GUI display)
plt.figure(figsize=(8, 5))
plt.bar(angles, num_matches, width=8, color='steelblue', edgecolor='k')
plt.title("Number of Matches vs Rotation Angle (BRIEF)")
plt.xlabel("Rotation Angle (degrees)")
plt.ylabel("Number of Matches")
plt.grid(True, linestyle="--", alpha=0.5)
hist_path = "../results/rotation_histogram.png"
plt.savefig(hist_path)
print(f"Saved histogram to {hist_path}")
