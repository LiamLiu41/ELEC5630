# Q2.1

import cv2
import matplotlib.pyplot as plt

from matchPics import matchPics

# Load the images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')

# Match the features
locs1, locs2 = matchPics(cv_cover, cv_desk)

# from matchPics import matches
# # Display the matched features
# match_result = cv2.drawMatches(cv_cover, locs1, cv_desk, locs2, matches[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('show all matchings', match_result)
# cv2.waitKey()

from matchPics import matches

match_result = cv2.drawMatches(cv_cover, locs1, cv_desk, locs2, matches[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('show all matchings', match_result)
# cv2.waitKey(0)
cv2.imwrite('../results/match_result.jpg', match_result)
print("Saved visualization to ../results/match_result.jpg")
