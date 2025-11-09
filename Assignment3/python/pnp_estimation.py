# -*- coding: utf-8 -*-
"""
Task 5: Perspective-n-Point (PnP) camera pose estimation
"""

import cv2
import numpy as np


def estimate_pose_pnp(points3d, points2d, K):
    """
    Estimate camera pose using PnP with RANSAC.
    Args:
        points3d: Nx3 3D points (from previous triangulation)
        points2d: Nx2 2D points (corresponding pixels in new image)
        K: 3x3 intrinsic matrix
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        inliers: inlier mask
    """
    # Convert to required shapes
    points3d = points3d.reshape(-1, 1, 3).astype(np.float32)
    points2d = points2d.reshape(-1, 1, 2).astype(np.float32)

    # Solve PnP with RANSAC
    # success, rvec, tvec, inliers = cv2.solvePnPRansac(
    #     points3d, points2d, K, None,
    #     reprojectionError=8.0, iterationsCount=1000, confidence=0.99
    # )

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d.astype(np.float32),
        points2d.astype(np.float32),
        K.astype(np.float32),
        None,
        flags=cv2.SOLVEPNP_ITERATIVE
    )


    if not success:
        raise RuntimeError("PnP failed to find a valid pose.")

    # Refine with Levenberg–Marquardt optimization
    rvec, tvec = cv2.solvePnPRefineLM(points3d, points2d, K, None, rvec, tvec)

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, inliers
