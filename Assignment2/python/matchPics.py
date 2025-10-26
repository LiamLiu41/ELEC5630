# import cv2

# def matchPics(I1, I2):
#     """_summary_

#     Args:
#         I1 (_type_): _description_
#         I2 (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
# #MATCHPICS Extract features, obtain their descriptors, and match them!

# ## Convert images to grayscale, if necessary

# ## Detect features in both images

# ## Obtain descriptors for the computed feature locations

# ## Match features using the descriptors
#     # return locs1, locs2
    
#     pass

import cv2
import numpy as np

# --- 模块级全局变量，用于保存匹配结果 ---
matches = []

def matchPics(I1, I2, ratio_test_thresh=0.75):
    """
    Detect features using FAST, describe them using BRIEF, and match them.
    Returns KeyPoint lists for drawMatches, and updates global 'matches'.
    """
    global matches

    # 灰度转换
    if len(I1.shape) == 3:
        gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = I1.copy()
    if len(I2.shape) == 3:
        gray2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = I2.copy()

    # FAST 特征检测
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    keypoints1 = fast.detect(gray1, None)
    keypoints2 = fast.detect(gray2, None)

    # BRIEF 描述子
    try:
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    except AttributeError:
        raise RuntimeError(
            "BRIEF descriptor not available. Please install opencv-contrib-python"
        )

    keypoints1, descriptors1 = brief.compute(gray1, keypoints1)
    keypoints2, descriptors2 = brief.compute(gray2, keypoints2)

    if descriptors1 is None or descriptors2 is None:
        matches = []
        return [], []

    # BFMatcher + Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_knn = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m_n in matches_knn:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio_test_thresh * n.distance:
            good_matches.append(m)

    # 更新全局变量
    matches = good_matches

    # 返回 KeyPoint 对象列表，供 drawMatches 使用
    return keypoints1, keypoints2
