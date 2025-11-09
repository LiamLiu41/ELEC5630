# -*- coding: utf-8 -*-
"""
Task 3: Essential matrix estimation and camera pose recovery
"""

import cv2
import numpy as np


def compute_essential_matrix(F, K1, K2):
    """
    Compute the essential matrix from fundamental matrix and camera intrinsics.
    """
    E = K2.T @ F @ K1
    # Enforce the singular value constraint: diag(1,1,0)
    U, S, Vt = np.linalg.svd(E)
    E_corrected = U @ np.diag([1, 1, 0]) @ Vt
    return E_corrected


def recover_pose_from_essential(E, pts1, pts2, K):
    """
    Recover relative camera pose (R, t) from essential matrix.
    Args:
        E: 3x3 essential matrix
        pts1, pts2: Nx2 matched normalized image points
        K: camera intrinsic matrix
    Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        mask: inlier mask from recoverPose
    """
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask
