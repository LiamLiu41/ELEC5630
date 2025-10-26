# # Q2.5

# import numpy as np
# import random

# def computeH_ransac(locs1, locs2):

#     pass

# Q2.5 Robust Homography Estimation using RANSAC

import numpy as np
import random
from computeH_norm import computeH_norm

def computeH_ransac(locs1, locs2, num_iter=1000, tol=3.0):
    """
    Estimate homography using RANSAC.

    Args:
        locs1: (N, 2) points in image 1
        locs2: (N, 2) corresponding points in image 2
        num_iter: number of RANSAC iterations
        tol: inlier pixel distance threshold

    Returns:
        bestH2to1: 3x3 homography matrix (from image 2 to image 1)
        inliers: Boolean array of size N indicating inlier matches
    """

    assert locs1.shape == locs2.shape, "locs1 and locs2 must have same shape"
    N = locs1.shape[0]

    bestH2to1 = None
    best_inliers = np.zeros(N, dtype=bool)
    max_inliers = 0

    for i in range(num_iter):
        # 1. Randomly select 4 correspondences
        idx = np.random.choice(N, 4, replace=False)
        x1_sample = locs1[idx]
        x2_sample = locs2[idx]

        # 2. Compute candidate Homography
        try:
            H_candidate = computeH_norm(x1_sample, x2_sample)
        except np.linalg.LinAlgError:
            continue

        # 3. Transform all locs2 points using H
        x2_h = np.hstack([locs2, np.ones((N, 1))])  # homogeneous
        x2_proj = (H_candidate @ x2_h.T).T
        x2_proj = x2_proj[:, :2] / x2_proj[:, [2]]

        # 4. Compute distances to locs1
        dists = np.linalg.norm(locs1 - x2_proj, axis=1)

        # 5. Determine inliers
        inliers = dists < tol
        num_inliers = np.sum(inliers)

        # 6. Update best model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            bestH2to1 = H_candidate

    # 7. Recompute H with all inliers
    if max_inliers >= 4:
        bestH2to1 = computeH_norm(locs1[best_inliers], locs2[best_inliers])
        bestH2to1 /= bestH2to1[-1, -1]

    return bestH2to1, best_inliers


# Optional Test
if __name__ == "__main__":
    np.random.seed(42)
    # Generate synthetic correspondences
    H_true = np.array([[1.2, 0.1, 100],
                       [0.05, 1.1, 50],
                       [0.0003, 0.0001, 1.0]])

    pts2 = np.random.rand(100, 2) * 100
    pts2_h = np.hstack([pts2, np.ones((100, 1))])
    pts1 = (H_true @ pts2_h.T).T
    pts1 = pts1[:, :2] / pts1[:, [2]]

    # Add noise + outliers
    pts1 += np.random.randn(*pts1.shape) * 2
    pts1[:10] += np.random.rand(10, 2) * 100  # 10 outliers

    # Estimate via RANSAC
    H_est, inliers = computeH_ransac(pts1, pts2, num_iter=5000, tol=4)
    print(f"Estimated H:\n{H_est}")
    print(f"Inliers: {np.sum(inliers)} / 100")
