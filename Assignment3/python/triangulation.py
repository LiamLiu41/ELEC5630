# -*- coding: utf-8 -*-
"""
Triangulation utilities for Task 4 of SfM project.

Provides functions to triangulate matched points given camera intrinsics and poses,
filter by cheirality and reprojection error, and save results as numpy/PLY.
"""

import numpy as np
import cv2
import os


def build_projection_matrix(K, R, t):
    """
    Build 3x4 projection matrix P = K * [R | t]
    Args:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    Returns:
        P: 3x4 projection matrix
    """
    Rt = np.hstack((R, t.reshape(3, 1)))
    P = K @ Rt
    return P


def triangulate_points(K, R, t, pts1, pts2):
    """
    Triangulate 3D points from matched 2D points in two views.
    Args:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation from cam1 to cam2
        t: 3x1 translation from cam1 to cam2
        pts1: Nx2 array in image1 (pixel coords)
        pts2: Nx2 array in image2 (pixel coords)
    Returns:
        points_3d: Nx3 array of 3D points in camera1 coordinate frame
        homog: 4xN homogeneous points returned by cv2.triangulatePoints (before division)
    """
    # Camera 1 is at identity pose [I | 0]
    P1 = build_projection_matrix(K, np.eye(3), np.zeros(3))
    P2 = build_projection_matrix(K, R, t.flatten())

    # cv2.triangulatePoints expects points as 2xN float
    pts1_t = pts1.T.astype(np.float64)
    pts2_t = pts2.T.astype(np.float64)

    homog = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)  # 4 x N
    # convert to non-homogeneous 3D points
    pts3d = homog[:3, :] / np.clip(homog[3, :], 1e-12, None)
    pts3d = pts3d.T  # N x 3
    return pts3d, homog


def filter_points_by_cheirality_and_reproj(K, R, t, pts3d, pts1, pts2, reproj_thresh=4.0):
    """
    Filter triangulated points by checking depth in both camera frames and reprojection error.
    Args:
        K, R, t: intrinsics and relative pose (cam1->cam2)
        pts3d: Nx3 points in cam1 frame
        pts1, pts2: Nx2 original image points
        reproj_thresh: pixel threshold for reprojection error
    Returns:
        mask: boolean array Nx indicating valid points
        errors: reprojection errors (max of two views) for each point
    """
    N = pts3d.shape[0]
    mask = np.zeros(N, dtype=bool)
    errors = np.full(N, np.inf, dtype=np.float64)

    # P1 = K[I|0], P2 = K[R|t]
    P1 = build_projection_matrix(K, np.eye(3), np.zeros(3))
    P2 = build_projection_matrix(K, R, t.flatten())

    for i in range(N):
        X = pts3d[i].reshape(3, 1)
        # depth in cam1: Z coordinate in cam1 frame (since pts3d are in cam1 frame)
        z1 = X[2, 0]
        # transform to cam2 frame: X2 = R * X + t
        X2 = R @ X + t.reshape(3, 1)
        z2 = X2[2, 0]

        if z1 <= 1e-6 or z2 <= 1e-6:
            mask[i] = False
            continue

        # reprojection to image1
        x1_proj_h = P1 @ np.vstack((X, [1.0]))  # 3x1
        x1_proj = (x1_proj_h[:2] / x1_proj_h[2]).ravel()
        err1 = np.linalg.norm(x1_proj - pts1[i])

        # reprojection to image2
        x2_proj_h = P2 @ np.vstack((X, [1.0]))
        x2_proj = (x2_proj_h[:2] / x2_proj_h[2]).ravel()
        err2 = np.linalg.norm(x2_proj - pts2[i])

        err = max(err1, err2)
        errors[i] = err
        if err <= reproj_thresh:
            mask[i] = True

    return mask, errors


def write_points_to_ply(points, out_path, colors=None):
    """
    Write Nx3 points (and optional colors Nx3 uint8) to an ASCII PLY file.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    N = points.shape[0]
    with open(out_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(N):
            x, y, z = points[i]
            if colors is None:
                f.write(f"{x} {y} {z}\n")
            else:
                r, g, b = colors[i]
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
    return out_path


def triangulate_and_filter(K, R, t, pts1, pts2, reproj_thresh=5.0, return_indices=False):
    """
    Triangulate points and filter by reprojection error.
    如果 return_indices=True，会返回每个有效 3D 点对应的 pts2 索引。
    """
    # 计算投影矩阵
    P1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K @ np.hstack((R, t.reshape(3,1)))

    pts4d_h = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d_h[:3] / pts4d_h[3]).T

    # 计算重投影误差
    pts2_proj = (P2 @ np.hstack((pts3d, np.ones((pts3d.shape[0],1)))).T).T
    pts2_proj = pts2_proj[:,:2] / pts2_proj[:,2,np.newaxis]
    errors = np.linalg.norm(pts2 - pts2_proj, axis=1)

    valid_mask = errors < reproj_thresh
    pts3d_valid = pts3d[valid_mask]

    if return_indices:
        triangulated_indices = np.where(valid_mask)[0]
        return pts3d_valid, valid_mask, triangulated_indices, errors
    else:
        return pts3d_valid, valid_mask, errors

