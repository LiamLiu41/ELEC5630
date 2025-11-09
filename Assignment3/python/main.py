# -*- coding: utf-8 -*-
"""
Multi-frame SfM for LLFF and templering datasets.
Supports Feature Matching, Fundamental/Essential Matrix, Triangulation, and accumulates camera poses.
Camera poses are saved with pickle to handle variable shapes.
"""

import argparse
import cv2
import numpy as np
import os
import glob
import pickle
import kagglehub
from feature_matching import (
    create_detector,
    detect_and_compute,
    match_descriptors,
    draw_and_save_matches,
    extract_matched_keypoints
)
from fundamental_matrix import estimate_fundamental_matrix
from pose_estimation import compute_essential_matrix, recover_pose_from_essential
from triangulation import triangulate_and_filter, write_points_to_ply
from bundle_adjustment import run_bundle_adjustment  
from non_linear_opt import refine_pnp_with_cv, refine_points_via_gauss_newton


# === Task 1: Dataset Loading & Initialization ===
def load_images_from_dataset(dataset_name, scene=None):
    dataset_name = dataset_name.lower()
    if dataset_name == "templering":
        base_dir = os.path.join("..", "data", "templeRing")
        img_paths = sorted(glob.glob(os.path.join(base_dir, "*.jpg")) +
                           glob.glob(os.path.join(base_dir, "*.JPG")) +
                           glob.glob(os.path.join(base_dir, "*.png")))
        if len(img_paths) < 2:
            raise RuntimeError(f"Not enough images in {base_dir}. Found {len(img_paths)}.")
        return img_paths

    elif dataset_name == "llff":
        print("Downloading LLFF dataset from KaggleHub (if not cached)...")
        llff_root = kagglehub.dataset_download("arenagrenade/llff-dataset-full")

        if scene is None:
            raise RuntimeError("You must specify a scene name for LLFF (e.g., fern or trex).")

        scene_dir = next((root for root, dirs, files in os.walk(llff_root)
                          if os.path.basename(root).lower() == scene.lower()), None)

        if scene_dir is None:
            raise RuntimeError(f"Scene path not found anywhere under {llff_root}")

        image_dir = os.path.join(scene_dir, "images")
        if not os.path.exists(image_dir):
            alt_dir = os.path.join(scene_dir, "images_2")
            if os.path.exists(alt_dir):
                image_dir = alt_dir
            else:
                raise RuntimeError(f"No image directory found under: {scene_dir}")

        print(f"Loading LLFF scene from: {image_dir}")
        img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                           glob.glob(os.path.join(image_dir, "*.jpg")) +
                           glob.glob(os.path.join(image_dir, "*.JPG")))
        if len(img_paths) < 2:
            raise RuntimeError(f"Not enough images found in scene {scene}")
        return img_paths

    else:
        raise RuntimeError(f"Unknown dataset name: {dataset_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-frame SfM for LLFF/templering datasets")
    parser.add_argument('--dataset', default='templering', choices=['templering', 'llff'], help='Dataset name')
    parser.add_argument('--scene', default=None, help='Scene name for LLFF (e.g., fern)')
    parser.add_argument('--detector', default='SIFT', help='Detector: SIFT | ORB | AKAZE')
    parser.add_argument('--ratio', type=float, default=0.75, help="Lowe's ratio threshold")
    parser.add_argument('--out_dir', default='results/task1', help='Output directory')
    parser.add_argument('--reproj_thresh', type=float, default=10.0, help='Triangulation reprojection threshold')
    return parser.parse_args()


def load_and_check_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    return img


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # === Task 1: Dataset Loading & Initialization ===
    print("\n========== Task 1: Dataset Loading & Initialization ==========")
    img_paths = load_images_from_dataset(args.dataset, args.scene)
    print(f"Found {len(img_paths)} images.")

    # Load intrinsic parameters
    K_path = os.path.join("..", "data", "templeRing", "camera.txt")
    K = np.loadtxt(K_path)
    print("Loaded intrinsic matrix K:\n", K)
    print("✅ Task 1 completed.\n")

    detector, is_binary = create_detector(args.detector)

    # Initialize first frame
    all_points3d = []
    camera_poses = []  # store (R, t) tuples
    prev_img = load_and_check_image(img_paths[0])
    prev_kp, prev_des = detect_and_compute(prev_img, detector)
    camera_poses.append((np.eye(3), np.zeros((3, 1))))  # first frame at origin

    # === Multi-frame Loop ===
    for i in range(1, len(img_paths)):
        print(f"\n========== Processing Frame {i+1}/{len(img_paths)} ==========")

        curr_img = load_and_check_image(img_paths[i])
        curr_kp, curr_des = detect_and_compute(curr_img, detector)

        if curr_des is None or prev_des is None or len(curr_kp) == 0 or len(prev_kp) == 0:
            print(f"No keypoints found in frame {i}, skipping.")
            prev_img, prev_kp, prev_des = curr_img, curr_kp, curr_des
            continue

        # === Task 2.2.1: Feature Correspondence ===
        matches = match_descriptors(prev_des, curr_des, binary_descriptor=is_binary, ratio_thresh=args.ratio)
        print(f"Found {len(matches)} good matches")
        pts1, pts2 = extract_matched_keypoints(prev_kp, curr_kp, matches)
        print("✅ Task 2.2.1: Feature Correspondence completed.")

        # === Task 2.2.2: Fundamental Matrix Estimation ===
        F, mask = estimate_fundamental_matrix(pts1, pts2)
        inlier_idx = mask.ravel().astype(bool)
        pts1_in, pts2_in = pts1[inlier_idx], pts2[inlier_idx]
        print(f"Estimated Fundamental Matrix:\n{F}")
        print("✅ Task 2.2.2: Fundamental Matrix Estimation completed.")

        # === Task 2.3.1: Essential Matrix Estimation ===
        E = compute_essential_matrix(F, K, K)
        print("Computed Essential Matrix:\n", E)
        print("✅ Task 2.3.1: Essential Matrix Estimation completed.")

        # === Task 2.3.2: Camera Pose Extraction ===
        R, t, pose_mask = recover_pose_from_essential(E, pts1_in, pts2_in, K)
        print("Recovered camera pose (R, t):\n", R, "\n", t)
        print("✅ Task 2.3.2: Camera Pose Extraction completed.")

        # === Task 2.4: Triangulation ===
        pts3d_valid, valid_mask, errors = triangulate_and_filter(
            K, R, t, pts1_in, pts2_in, reproj_thresh=args.reproj_thresh
        )
        print(f"Triangulated {len(pts1_in)} points, kept {pts3d_valid.shape[0]} valid points")
        all_points3d.append(pts3d_valid)
        camera_poses.append((R, t))
        print("✅ Task 2.4: Triangulation completed.")

        # === ✅ Task 2.7: Non-linear Optimization ===
        print("\n========== Task 2.7: Non-linear Optimization ==========")
        if pts3d_valid.shape[0] < 3:
            print("Not enough valid 3D points for non-linear optimization, skipping this frame.")
        else:
            pts3d_refined = refine_points_via_gauss_newton(
                pts3d_valid, pts1_in[valid_mask], pts2_in[valid_mask], K, R, t
            )
            R_refined, t_refined = refine_pnp_with_cv(pts3d_refined, pts2_in[valid_mask], K, R, t)
            print("Refined with non-linear optimization.")
            all_points3d[-1] = pts3d_refined
            camera_poses[-1] = (R_refined, t_refined)
        print("✅ Task 2.7: Non-linear Optimization completed.")

        # === Task 2.6: Bundle Adjustment ===
        print("\n========== Task 2.6: Bundle Adjustment ==========")
        camera_params = []
        for R_, t_ in camera_poses:
            rvec, _ = cv2.Rodrigues(R_)
            camera_params.append(np.hstack((rvec.flatten(), t_.flatten())))
        camera_params = np.array(camera_params)
        points_3d_concat = np.vstack(all_points3d)

        n_cams = len(camera_params)
        n_points = points_3d_concat.shape[0]
        camera_indices = np.repeat(np.arange(n_cams), n_points // n_cams)
        point_indices = np.tile(np.arange(n_points // n_cams), n_cams)
        points_2d = np.vstack([pts1_in[:n_points // n_cams], pts2_in[:n_points // n_cams]])

        optimized_cameras, optimized_points = run_bundle_adjustment(
            camera_params, points_3d_concat, camera_indices, point_indices, points_2d, K
        )
        print("✅ Task 2.6: Bundle Adjustment completed.\n")

        # Update previous frame
        prev_img, prev_kp, prev_des = curr_img, curr_kp, curr_des

    # === Task 5: Multi-frame Fusion & Saving ===
    print("\n========== Task 5: Multi-frame Fusion & Saving ==========")
    if all_points3d:
        points3d_array = np.vstack(all_points3d)
        write_points_to_ply(points3d_array, os.path.join(args.out_dir, "points3d_all_frames.ply"))
        np.save(os.path.join(args.out_dir, "points3d_all_frames.npy"), points3d_array)
        print(f"Saved {points3d_array.shape[0]} total 3D points.")
    print("✅ Task 5: Multi-frame Fusion completed.")

    # Save camera poses with pickle
    with open(os.path.join(args.out_dir, "camera_poses.pkl"), "wb") as f:
        pickle.dump(camera_poses, f)
    print(f"Saved camera poses for {len(camera_poses)} frames as pickle file.")
    print("✅ Camera pose saving completed.")

    print("\n========== 🎉 All Tasks Completed Successfully ==========")


if __name__ == '__main__':
    main()
