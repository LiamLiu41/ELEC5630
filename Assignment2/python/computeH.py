# # Q2.3

# import numpy as np
# import cv2 

# def computeH(x1, x2):
#     pass


# Q2.3 Homography Computation (DLT)
import numpy as np
import cv2

def computeH(x1, x2):
    """
    Compute the homography matrix H2to1 that maps x2 -> x1 using DLT.

    Args:
        x1: (N, 2) array of points in image 1
        x2: (N, 2) array of corresponding points in image 2

    Returns:
        H2to1: (3, 3) homography matrix, normalized so that H[2,2] = 1
    """

    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"
    N = x1.shape[0]
    assert N >= 4, "At least 4 points are required to compute a homography"

    # Construct matrix A (2N x 9)
    A = []
    for i in range(N):
        x, y = x2[i, 0], x2[i, 1]
        u, v = x1[i, 0], x1[i, 1]
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.array(A)

    # Solve Ah = 0 using SVD
    # A = U Σ V^T  => h = last column of V (or last row of V^T)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]    # smallest singular value
    H2to1 = h.reshape(3, 3)

    # Normalize so that H[2,2] = 1
    H2to1 = H2to1 / H2to1[-1, -1]

    return H2to1


# Optional test example
if __name__ == "__main__":
    # Four corresponding points (square -> quadrilateral)
    x1 = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    x2 = np.array([[0.2, 0.1], [0.9, -0.1], [1.2, 1.1], [0, 0.9]])
    H = computeH(x1, x2)
    print("Computed H:\n", H)

    # Verify mapping
    x2_h = np.hstack([x2, np.ones((x2.shape[0], 1))])
    x1_pred = (H @ x2_h.T).T
    x1_pred = x1_pred[:, :2] / x1_pred[:, [2]]
    print("Reprojected points:\n", x1_pred)
