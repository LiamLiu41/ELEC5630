# # Q2.4

# import numpy as np

# # !!! YOU CAN USE np.linalg.svd()

# def computeH_norm(x1, x2):
#     # Compute centroids of the points
#     centroid1 = None
#     centroid2 = None

#     # Shift the origin of the points to the centroid

#     # Normalize the points so that the average distance from the origin is equal to sqrt(2)


#     # Similarity transform 1


#     # Similarity transform 2


#     # Compute Homography

#     # Denormalization
#     H2to1 = None

#     return H2to1

# Q2.4 Homography Normalization
import numpy as np
from computeH import computeH  # 调用上一题的 DLT 实现

def computeH_norm(x1, x2):
    """
    Compute the normalized homography matrix H2to1 that maps x2 -> x1.

    Args:
        x1: (N, 2) array of points in image 1
        x2: (N, 2) array of corresponding points in image 2

    Returns:
        H2to1: (3, 3) homography matrix (denormalized)
    """

    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    N = x1.shape[0]

    # ---------- Normalize x1 ----------
    centroid1 = np.mean(x1, axis=0)
    shifted1 = x1 - centroid1
    avg_dist1 = np.mean(np.sqrt(np.sum(shifted1**2, axis=1)))
    scale1 = np.sqrt(2) / avg_dist1

    T1 = np.array([
        [scale1, 0, -scale1 * centroid1[0]],
        [0, scale1, -scale1 * centroid1[1]],
        [0, 0, 1]
    ])
    x1_norm = (T1 @ np.vstack((x1.T, np.ones(N)))).T[:, :2]

    # ---------- Normalize x2 ----------
    centroid2 = np.mean(x2, axis=0)
    shifted2 = x2 - centroid2
    avg_dist2 = np.mean(np.sqrt(np.sum(shifted2**2, axis=1)))
    scale2 = np.sqrt(2) / avg_dist2

    T2 = np.array([
        [scale2, 0, -scale2 * centroid2[0]],
        [0, scale2, -scale2 * centroid2[1]],
        [0, 0, 1]
    ])
    x2_norm = (T2 @ np.vstack((x2.T, np.ones(N)))).T[:, :2]

    # ---------- Compute H' using normalized coordinates ----------
    H_norm = computeH(x1_norm, x2_norm)

    # ---------- Denormalize ----------
    H2to1 = np.linalg.inv(T1) @ H_norm @ T2
    H2to1 = H2to1 / H2to1[-1, -1]

    return H2to1


# Optional test
if __name__ == "__main__":
    x1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * 500
    x2 = np.array([[50, 30], [480, 20], [520, 520], [40, 550]])
    H = computeH_norm(x1, x2)
    print("Normalized Homography H2to1:\n", H)

    # Check mapping
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])
    x1_pred = (H @ x2_h.T).T
    x1_pred = x1_pred[:, :2] / x1_pred[:, [2]]
    print("Reprojected points:\n", x1_pred)
