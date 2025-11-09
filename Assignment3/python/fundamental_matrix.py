# -*- coding: utf-8 -*-
"""
Task 2: Fundamental matrix estimation with RANSAC
"""

import cv2
import numpy as np
import os


def estimate_fundamental_matrix(pts1, pts2, ransac_thresh=0.5, confidence=0.99):
    """
    Estimate fundamental matrix using RANSAC and return inliers.
    Args:
        pts1, pts2: Nx2 arrays of matched keypoints (float32)
        ransac_thresh: reprojection threshold in pixels
        confidence: RANSAC confidence level
    Returns:
        F: 3x3 fundamental matrix
        mask: Nx1 inlier mask (1 for inliers)
    """
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=confidence
    )
    if F is None:
        raise RuntimeError("Fundamental matrix estimation failed.")
    return F, mask


def save_inlier_matches(img1, img2, pts1, pts2, mask, out_path):
    """
    Visualize inlier matches after RANSAC filtering.
    """
    inlier_idx = mask.ravel().astype(bool)
    pts1_in = pts1[inlier_idx]
    pts2_in = pts2[inlier_idx]
    # Convert points back to cv2.KeyPoint for visualization
    kp1 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=3) for p in pts1_in]
    kp2 = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=3) for p in pts2_in]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]

    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, matched_img)
    return out_path
