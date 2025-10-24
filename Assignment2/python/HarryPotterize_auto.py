# Q2.6

import cv2
import numpy as np

from .matchPics import matchPics
from .computeH_ransac import computeH_ransac
from .warpH import warpH
from .compositeH import compositeH

# Load images
cv_img = cv2.imread('../data/cv_cover.jpg')
desk_img = cv2.imread('../data/cv_desk.png')
hp_img = cv2.imread('../data/hp_cover.jpg')

# Extract features and match
locs1, locs2 = matchPics(cv_img, desk_img)

# Compute homography using RANSAC
bestH2to1, _ = computeH_ransac(locs1, locs2);

# Scale harry potter image to template size
scaled_hp_img = cv2.resize(hp_img, (desk_img.shape[1], desk_img.shape[0]))

# Display warped image
warped_hp_img = warpH(scaled_hp_img, bestH2to1, desk_img.shape)
cv2.imshow('Warped Image', warped_hp_img)
cv2.waitKey(0)

# Display composite image
composite_img = compositeH(bestH2to1, scaled_hp_img, desk_img)
cv2.imshow('Composite Image', composite_img)
cv2.waitKey(0)
cv2.destroyAllWindows()