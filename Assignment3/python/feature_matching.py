# -*- coding: utf-8 -*-
"""
Feature matching utilities for Task 1 of SfM project.
"""

import cv2
import numpy as np
import os


def create_detector(detector_name="SIFT"):
    """
    Create feature detector and descriptor extractor.
    Returns (detector, is_binary_descriptor)
    """
    name = detector_name.upper()
    if name == "SIFT":
        # SIFT gives float descriptors -> use L2
        try:
            detector = cv2.SIFT_create()
        except AttributeError:
            # fallback if SIFT not available
            detector = cv2.ORB_create(nfeatures=5000)
            return detector, True
        return detector, False
    elif name == "ORB":
        detector = cv2.ORB_create(nfeatures=5000)
        return detector, True
    elif name == "AKAZE":
        detector = cv2.AKAZE_create()
        # AKAZE descriptors are binary
        return detector, True
    else:
        # Default fallback
        detector = cv2.ORB_create(nfeatures=5000)
        return detector, True


def match_descriptors(desc1, desc2, binary_descriptor=True, ratio_thresh=0.75):
    """
    Match descriptors with KNN and apply Lowe's ratio test.
    Returns list of good matches (cv2.DMatch objects from knn).
    """
    # choose norm type
    if binary_descriptor:
        norm = cv2.NORM_HAMMING
    else:
        norm = cv2.NORM_L2

    # BFMatcher with crossCheck=False (we do ratio test)
    bf = cv2.BFMatcher(norm, crossCheck=False)
    # knnMatch to get two nearest neighbors
    knn_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m_n in knn_matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


def detect_and_compute(img, detector):
    """
    Detect keypoints and compute descriptors using provided detector.
    img: grayscale or color image (will convert to gray if color)
    returns: keypoints, descriptors
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    kp, des = detector.detectAndCompute(gray, None)
    return kp, des


def draw_and_save_matches(img1, img2, kp1, kp2, matches, save_path):
    """
    Draw matches between two images and save to disk.
    Compatible with OpenCV >= 4.12
    """
    if not matches or len(kp1) == 0 or len(kp2) == 0:
        print("No matches or invalid keypoints, skipping visualization.")
        return

    # draw matches with default thickness
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        matchesMask=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchesThickness=1  # explicitly set
    )
    cv2.imwrite(save_path, matched_img)



def extract_matched_keypoints(kp1, kp2, matches):
    """
    From keypoints and matches, return Nx2 arrays of point coordinates for image1 and image2.
    """
    pts1 = []
    pts2 = []
    for m in matches:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)
    return pts1, pts2
