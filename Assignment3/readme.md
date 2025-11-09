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

## ⚙️ Full Pipeline

The implemented Structure from Motion (SfM) pipeline consists of the following stages:

1. **Dataset Loading & Initialization**  
   The system first loads all input images and their corresponding camera intrinsics (K and distortion coefficients).  
   The intrinsic parameters are directly read from the dataset, while extrinsics are estimated during reconstruction and later compared with ground truth for evaluation.

2. **Feature Extraction & Matching**  
   I implemented a flexible feature extraction module that supports both classical and learning-based detectors (e.g., SIFT, ORB, SuperGlue, LightGlue).  
   Keypoints are detected and described for each frame, and feature matching is performed between adjacent images to establish correspondences.

3. **Fundamental Matrix Estimation**  
   Matched keypoints are used to estimate the fundamental matrix **F**.  
   The implementation supports both the classic 8-point algorithm and RANSAC-based outlier rejection to improve robustness against mismatches.

4. **Essential Matrix Estimation & Camera Pose Recovery**  
   The essential matrix **E = Kᵀ F K** is computed and corrected by enforcing singular values (1, 1, 0).  
   Four possible pose configurations are then decomposed, and the correct one is selected using the **cheirality condition**, ensuring reconstructed points lie in front of both cameras.

5. **Triangulation**  
   Using the recovered relative pose, 3D scene points are reconstructed through triangulation (via DLT and OpenCV built-in functions).  
   This step provides an initial estimate of the scene structure.

6. **Perspective-n-Point (PnP)**  
   After obtaining initial 3D points, the next camera poses are estimated incrementally using the **PnP** algorithm based on new 2D–3D correspondences.  
   This step extends the reconstruction to multiple frames.

7. **Bundle Adjustment**  
   A bundle adjustment module is implemented to jointly refine all estimated camera parameters and 3D points by minimizing reprojection error.  
   The optimization is done using `scipy.optimize.least_squares`.  
   > **Note:** This step is computationally expensive.  
   > For quick reproduction, this part can be commented out or skipped.

8. **(Optional) Non-linear Optimization for Triangulation and PnP**  
   To further improve accuracy, optional non-linear refinements are applied after each **Triangulation** and **PnP** step.  
   These local optimizations minimize geometric projection error and help stabilize the reconstruction process.

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