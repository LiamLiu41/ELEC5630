# ELEC5630 Project 3: Structure from Motion

## Introduction
This project implements a complete pipeline for Multi-frame Structure from Motion (SfM) that supports feature matching, fundamental and essential matrix estimation, triangulation, and camera pose recovery.

## Code Logic
1. **Dataset Loading & Initialization**: Load the specified dataset images and camera intrinsics.
2. **Feature Extraction & Matching**: Use the specified feature detector to extract keypoints, describe features, and match keypoints between adjacent frames.
3. **Fundamental Matrix Estimation**: Estimate the fundamental matrix F from matched keypoints.
4. **Essential Matrix Estimation & Camera Pose Recovery**: Compute the essential matrix from the fundamental matrix and recover camera poses.
5. **Triangulation**: Perform triangulation on matched keypoints to generate a 3D point cloud.
6. **Bundle Adjustment**: Optimize camera parameters and 3D points to minimize reprojection error.
7. **Result Saving**: Save 3D point cloud and camera poses.

## Command Line Execution
```bash
python main.py --dataset llff --scene fern --out_dir results/task/fern
python main.py --dataset templering --detector SIFT --out_dir results/task
```

ps.
llff dataset dir of mine: /home/liuyu/.cache/kagglehub/datasets/arenagrenade/llff-dataset-full/versions/1/nerf_llff_data/fern/images/
templering dataset dir of mine: ../data/templeRing

results dir: ../python/results/task

## Result Display & Analysis
- After running, you will find the following files in the results/task directory:
- - points3d_all_frames.ply: Point cloud file containing all generated 3D points.
- - points3d_all_frames.npy: Numpy array saving all 3D points.
- - camera_poses.pkl: Pickle file saving camera poses.
### Result Analysis
1. 3D Point Cloud:
Open the points3d_all_frames.ply file with a point cloud viewer (like CloudCompare or MeshLab) to check the completeness and density of the point cloud.

Ensure the point cloud covers key areas of the reconstructed scene.
2. Camera Poses:
Load the camera_poses.pkl file to check the reasonableness of camera poses, ensuring the pose variations align with the scene geometry.

3. Reprojection Error:
Calculate reprojection error for each camera pose to evaluate the accuracy of the cameras and 3D points.
