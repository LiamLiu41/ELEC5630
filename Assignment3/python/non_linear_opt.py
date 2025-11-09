# -*- coding: utf-8 -*-
"""
Non-linear optimization for SfM refinement.
Includes:
1. Non-linear PnP refinement using OpenCV LM solver.
2. Non-linear Triangulation refinement using Gauss-Newton or Levenberg-Marquardt.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares


def refine_pnp_with_cv(points3D, points2D, K, R_init, t_init, method="LM"):
    """Non-linear refinement for PnP results using OpenCV Levenberg-Marquardt."""
    points3D = points3D.reshape(-1, 1, 3).astype(np.float32)
    points2D = points2D.reshape(-1, 1, 2).astype(np.float32)
    rvec, _ = cv2.Rodrigues(R_init)
    tvec = t_init.astype(np.float32)

    if method == "LM":
        rvec, tvec = cv2.solvePnPRefineLM(points3D, points2D, K, None, rvec, tvec)
    else:
        rvec, tvec = cv2.solvePnPRefineVVS(points3D, points2D, K, None, rvec, tvec)

    R_refined, _ = cv2.Rodrigues(rvec)
    t_refined = tvec.reshape(3, 1)
    return R_refined, t_refined


def project_point(X, K, R, t):
    """Project a single 3D point (3,) into 2D using camera intrinsics."""
    X = X.reshape(3, 1)               # (3,1)
    x_h = K @ (R @ X + t)             # (3,1)
    x = x_h[:2] / x_h[2]              # (2,1)
    return x.flatten()                # (2,)


def refine_points_via_gauss_newton(pts3D, pts1, pts2, K, R, t, max_iter=10):
    """
    Refine triangulated 3D points by minimizing reprojection error
    between two views using Gauss-Newton (Levenberg-Marquardt via SciPy).
    """

    def reproj_error(X, x1, x2, K, R, t):
        proj1 = project_point(X, K, np.eye(3), np.zeros((3, 1)))
        proj2 = project_point(X, K, R, t)
        return np.hstack((proj1 - x1, proj2 - x2))

    refined_pts = []
    for X, x1, x2 in zip(pts3D, pts1, pts2):
        X = np.asarray(X).flatten()
        x1 = np.asarray(x1).flatten()
        x2 = np.asarray(x2).flatten()

        # skip ill-conditioned points
        if np.any(np.isnan(X)) or np.any(np.isnan(x1)) or np.any(np.isnan(x2)):
            refined_pts.append(X)
            continue

        res = least_squares(
            reproj_error,
            X,
            args=(x1, x2, K, R, t),
            method='lm',
            max_nfev=max_iter
        )
        refined_pts.append(res.x)
    return np.array(refined_pts)
