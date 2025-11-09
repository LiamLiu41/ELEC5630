# ELEC5630 (L1) Project 3: Structure from Motion

---

## Introduction

This project implements a **Multi-frame Structure from Motion (SfM)** pipeline to reconstruct a 3D point cloud and recover camera poses from multiple images.  

The complete pipeline includes:
- Feature extraction and matching  
- Fundamental & essential matrix estimation  
- Camera pose recovery  
- Triangulation  
- Perspective-n-Point (PnP)  
- Bundle adjustment  
- (Optional) Non-linear optimization for refinement  

---

## Full Pipeline

1. **Dataset Loading & Initialization**  
   Load dataset images and known intrinsic parameters (K, distortion coefficients).  
   Extrinsic parameters must **not** be used; instead, estimated poses should be evaluated against ground truth.

2. **Feature Extraction & Matching**  
   Detect keypoints and compute descriptors using OpenCV (e.g., SIFT, ORB), SuperGlue, or LightGlue.  
   Perform feature matching between adjacent frames to obtain correspondences.

3. **Fundamental Matrix Estimation**  
   Estimate the fundamental matrix **F** using the 8-point algorithm or RANSAC to remove outliers.

4. **Essential Matrix Estimation & Camera Pose Recovery**  
   Compute **E = Kᵀ F K**, enforce singular values (1, 1, 0), and decompose to obtain four possible poses.  
   Use the **cheirality condition** to select the correct one.

5. **Triangulation**  
   Triangulate matched keypoints to obtain initial 3D points using DLT or OpenCV built-in functions.

6. **Perspective-n-Point (PnP)**  
   Estimate new camera poses incrementally using previously triangulated 3D points and 2D correspondences.

7. **Bundle Adjustment (Task 2.6)**  
   Refine all camera poses and 3D points jointly by minimizing reprojection error using non-linear optimization (e.g., `scipy.optimize.least_squares`).  
   > **Note:** This step can be **computationally expensive**.  
   > If you wish to **quickly reproduce results**, you can **comment out** or **skip** this part.

8. **(Optional) Non-linear Optimization for Triangulation and PnP**  
   Further refine the results of **triangulation** and **PnP** by minimizing projection error after each step.  
   Implementing this step can grant **+20% bonus points** for improved reconstruction accuracy.

---

## Command Line Execution

```bash
# Run on LLFF dataset (fern scene)
python main.py --dataset llff --scene fern --out_dir results/task/fern

# Run on TempleRing dataset
python main.py --dataset templering --detector SIFT --out_dir results/task
```

## Dataset Paths

- LLFF Dataset (Fern Scene)
/home/liuyu/.cache/kagglehub/datasets/arenagrenade/llff-dataset-full/versions/1/nerf_llff_data/fern/images/

- TempleRing Dataset
../data/templeRing

- Results Directory
../python/results/task