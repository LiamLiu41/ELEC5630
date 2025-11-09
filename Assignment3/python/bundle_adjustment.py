# -*- coding: utf-8 -*-
"""
Task 2.6: Bundle Adjustment (BA)
Refine camera poses and 3D points by minimizing reprojection error.
"""

import numpy as np
from scipy.optimize import least_squares
import cv2


def project(points_3d, rvec, tvec, K):
    """Project 3D points to 2D using camera intrinsics."""
    pts_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    return pts_2d.reshape(-1, 2)


def bundle_adjustment_error(params, n_cams, n_points, camera_indices, point_indices, points_2d, K):
    """Compute reprojection residuals for all observations."""
    camera_params = params[:n_cams * 6].reshape((n_cams, 6))
    points_3d = params[n_cams * 6:].reshape((n_points, 3))

    residuals = []
    for cam_idx, pt_idx, obs in zip(camera_indices, point_indices, points_2d):
        rvec = camera_params[cam_idx, :3]
        tvec = camera_params[cam_idx, 3:].reshape(3, 1)
        proj = project(points_3d[pt_idx:pt_idx + 1], rvec, tvec, K)
        residuals.append((proj.ravel() - obs))
    return np.concatenate(residuals)


def run_bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d, K):
    """Run non-linear least squares optimization to minimize reprojection error."""
    n_cams = camera_params.shape[0]
    n_points = points_3d.shape[0]
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    print(f"Running bundle adjustment with {n_cams} cameras and {n_points} 3D points...")

    res = least_squares(
        bundle_adjustment_error, x0,
        verbose=1, x_scale='jac', ftol=1e-4, method='trf',
        args=(n_cams, n_points, camera_indices, point_indices, points_2d, K)
    )

    optimized_cameras = res.x[:n_cams * 6].reshape((n_cams, 6))
    optimized_points = res.x[n_cams * 6:].reshape((n_points, 3))
    return optimized_cameras, optimized_points
