# ELEC HW2

## **2.1 Feature Detection, Description, and Matching**

```bash
cd python
python3 main.py
```

### Description:

FAST + BRIEF

Both images (`cv_cover.jpg` and `cv_desk.png`) are first converted to grayscale. FAST detects corner-like features, and BRIEF extracts binary descriptors for each keypoint.

Feature pairs are matched using a **Brute-Force Matcher** with Hamming distance, followed by **Lowe’s ratio test (0.75)** to remove ambiguous matches.

### Results:

../results/match_result.jpg

![c64899e5c5cd491a08e3461ebf83bcd8.png](ELEC%20HW2/c64899e5c5cd491a08e3461ebf83bcd8.png)

Most matches are correctly aligned along the book's edges and text, confirming successful feature correspondence.

## 2.2 **BRIEF and Rotation**

```python
python3 briefRotTest.py
```

### Description:

We evaluate the **rotation robustness** of the BRIEF descriptor.

The cover image is rotated from 0° to 360° in 10° increments, and each rotated version is matched against the original using the same **FAST + BRIEF** pipeline.

For each rotation, the number of valid matches (after Lowe’s ratio test) is recorded and plotted as a histogram.

### Results:

Outputs:

```bash
Rotation   0° -> 1548 matches
Saved match visualization at 0° to ../results/match_rotation_0.png
Rotation  10° ->  622 matches
Rotation  20° ->   57 matches
Rotation  30° ->   30 matches
Rotation  40° ->   16 matches
Rotation  50° ->   15 matches
Rotation  60° ->   17 matches
Rotation  70° ->   15 matches
Rotation  80° ->    8 matches
Rotation  90° ->   15 matches
Saved match visualization at 90° to ../results/match_rotation_90.png
Rotation 100° ->   18 matches
Rotation 110° ->   17 matches
Rotation 120° ->   20 matches
Rotation 130° ->   10 matches
Rotation 140° ->   14 matches
Rotation 150° ->   24 matches
Rotation 160° ->   16 matches
Rotation 170° ->   23 matches
Rotation 180° ->   27 matches
Saved match visualization at 180° to ../results/match_rotation_180.png
Rotation 190° ->   22 matches
Rotation 200° ->   13 matches
Rotation 210° ->   17 matches
Rotation 220° ->   15 matches
Rotation 230° ->   20 matches
Rotation 240° ->   14 matches
Rotation 250° ->   15 matches
Rotation 260° ->   26 matches
Rotation 270° ->   14 matches
Rotation 280° ->   17 matches
Rotation 290° ->   16 matches
Rotation 300° ->   17 matches
Rotation 310° ->   19 matches
Rotation 320° ->   15 matches
Rotation 330° ->   25 matches
Rotation 340° ->   44 matches
Rotation 350° ->  583 matches
Rotation 360° -> 1548 matches
Saved histogram to ../results/rotation_histogram.png
```

match_rotation_0.png:

![dcefa3c9df7d7816cfe64b76858ba514.png](ELEC%20HW2/dcefa3c9df7d7816cfe64b76858ba514.png)

match_rotation_90.png:

![fcd5589196401cb1ea2ce2c68e931c7f.png](ELEC%20HW2/fcd5589196401cb1ea2ce2c68e931c7f.png)

match_rotation_180.png:

![04319025bf464e36e39abbd0a37fd0b3.png](ELEC%20HW2/04319025bf464e36e39abbd0a37fd0b3.png)

rotation_histogram.png:

![da0528d507ff715130d21c02fb53d264.png](ELEC%20HW2/da0528d507ff715130d21c02fb53d264.png)

The results show a steep decline in match count as rotation increases—dropping from ≈1500 matches at 0° to under 30 around 90° – 180°.

Match numbers then rise again near 350° and 360°, where the orientation approaches the original.

This demonstrates that **BRIEF is not rotation-invariant**, since its binary intensity comparisons depend on fixed sampling patterns relative to pixel orientation.

## **2.3 Homography Computation**

```python
python computeH.py
```

### Description:

In this task, we implement the **Direct Linear Transform (DLT)** algorithm to compute the homography matrix ( `$H_{2\to1}$` ) between two sets of corresponding points.

For each point pair (`$(x_1, y_1)$`) and ($(x_2, y_2)$), two linear equations are constructed to form a system (A h = 0), where ($h$) is the 9-element vectorized form of ($H$).

The matrix ($H$) is then obtained as the **right singular vector** corresponding to the smallest singular value of ($A$) (via SVD), and normalized so that ($H_{33} = 1$).

### Results:

```bash
Computed H:
 [[ 2.80963692  0.70240923 -0.63216831]
 [ 0.7804547   2.73159145 -0.42925008]
 [ 1.04513064  1.1435358   1.        ]]
Reprojected points:
 [[3.35572026e-16 1.25839510e-16]
 [1.00000000e+00 3.64752202e-16]
 [1.00000000e+00 1.00000000e+00]
 [1.09425661e-16 1.00000000e+00]]
```

The computed homography correctly maps all source points to their destinations, as shown by the **reprojection results** (errors ≈ 1e-16).

This verifies the correctness and numerical stability of the DLT implementation, providing a solid foundation for later normalization and RANSAC refinement steps.

## **2.4 Homography Normalization**

```python
python computeH_norm.py
```

### Description

In this task, we extend the DLT algorithm by applying **coordinate normalization** before computing the homography.

Each point set is translated so that its centroid lies at the origin and scaled such that the **average distance to the origin equals $\sqrt{2}$**

The normalized points are then used to compute a temporary homography ( $H'$ ) via the DLT method, and the final matrix is **denormalized** as

( $H = T_1^{-1} H' T_2$ ).

This normalization step improves numerical stability, especially when image coordinates are large.

### Results:

```bash
Normalized Homography H2to1:
 [[ 1.12930601e+00  2.17174232e-02 -5.71168230e+01]
 [ 2.51353463e-02  1.08081989e+00 -3.36813641e+01]
 [-7.04716974e-05  2.29761543e-04  1.00000000e+00]]
Reprojected points:
 [[-7.08156770e-14 -7.08156770e-15]
 [ 5.00000000e+02  1.02471342e-13]
 [ 5.00000000e+02  5.00000000e+02]
 [ 5.05926927e-14  5.00000000e+02]]

```

The normalized homography accurately maps all points to their corresponding locations, with reprojection errors near ($10^{-13}$).

This confirms the effectiveness of normalization in enhancing the **precision and stability** of homography estimation, which is crucial for later RANSAC-based refinement.

## **2.5 RANSAC**

```python
python computeH_ransac.py
```

### Description

In this task, we implement a **RANSAC-based** method to robustly estimate the homography matrix ( $H_{2\to1}$ ) in the presence of noisy correspondences and outliers.

During each iteration, four random point pairs are sampled to compute a candidate homography using the normalized DLT algorithm.

All remaining points are then projected using this candidate, and those with a reprojection error below a given tolerance (e.g., 3–5 pixels) are classified as **inliers**.

The model producing the **largest inlier set** is selected as the best estimate, and the final ( $H_{2\to1}$ ) is recomputed using only those inliers.

### Results

```bash
Estimated H:
[[1.21305338e+00 8.97575614e-02 1.00023412e+02]
 [5.24914396e-02 1.10090755e+00 4.96943897e+01]
 [3.46795715e-04 3.43497231e-05 1.00000000e+00]]
Inliers: 79 / 100
```

The estimated homography closely matches the ground truth, successfully filtering out 21 outliers and preserving 79 inliers.

This demonstrates that RANSAC effectively increases the robustness of homography estimation against mismatched or noisy feature correspondences, forming the foundation for accurate geometric alignment in the final image warping stage.

## **2.6 HarryPotterizing a Book**

```bash
python HarryPotterize_auto.py
```

### Description

We integrate all previous components into a complete **automatic image warping pipeline**.

The goal is to replace the cover of the book on the desk with the *Harry Potter* cover image using the computed homography.

### Results:

![b1bab14b203ed80840315da909d0b05b.png](ELEC%20HW2/b1bab14b203ed80840315da909d0b05b.png)

![b039737475d522ac240c2f56f902fa22.png](ELEC%20HW2/b039737475d522ac240c2f56f902fa22.png)

The final composite demonstrates that the estimated homography correctly aligns the planar regions between the desk and the cover.

The warped *Harry Potter* image appears geometrically consistent with the original scene, confirming the successful integration of feature detection, homography estimation, and image compositing.