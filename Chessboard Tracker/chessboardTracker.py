# __author__ = 'gauravhirlekar'
#
#
# import cv2
#
# cam = cv2.VideoCapture(0)
#
# while True:
#     img = cv2.cvtColor(cv2.resize(cam.read()[1], (640, 480)), cv2.COLOR_BGR2GRAY)
#     ret, corners = cv2.findChessboardCorners(img, (5, 4), None, cv2.CALIB_CB_ADAPTIVE_THRESH)
#     print ret
#     cv2.drawChessboardCorners(img, (5, 4), corners, ret)
#     cv2.imshow('', img)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

import numpy as np
import cv2
import glob

cam = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Arrays to store object points and image points from all the images.
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')

while True:
    img = cv2.resize(cam.read()[1], (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret = False
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (5, 4))
    # If found, add object points, image points (after refining them)
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
    cv2.drawChessboardCorners(img, (5, 4), corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey(10)
    # cv2.waitKey(0)
cv2.destroyAllWindows()
