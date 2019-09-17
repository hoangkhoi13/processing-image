#!/usr/bin/env python
"""Camera Calibration using OpenCV
Syntax: calib.py sample_folder ext num_row num_col (attempts) (accuracy)
E.g: calib.py sample_group_resized jpg 7 4 #row should be bigger than col
Images should be 680x480 for optimal result.
-h or --help for help.

Reference Documents:
https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
"""
import cv2 as cv
import numpy as np
import glob
import sys
import time

# Start timer
stt = time.time()

# Getting variables
if len(sys.argv) < 5:
    if '-h' in sys.argv or '--help' in sys.argv:
        print(
            """Camera Calibration using OpenCV
            Syntax: calib.py sample_folder ext num_row num_col (attempts) (accuracy)
            E.g: calib.py sample_group_resized jpg 7 4 #row should be bigger than col
            -h or --help for help.

            Reference Documents:
            https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
            https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
            https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
            """)
    else:
        print("\n Not enough arguments were provided!")
        print("-h or --help for help.")
    sys.exit()

else:
    sample_folder = str(sys.argv[1])
    ext = str(sys.argv[2])
    num_row = int(sys.argv[3])
    num_col = int(sys.argv[4])
    try:
        attempts = int(sys.argv[5])
    except IndexError:
        attempts = 30
    try:
        epsilon = int(sys.argv[6])
    except:
        epsilon = 0.001

globfiles = '{}/*.{}'.format(sample_folder, ext)
images = glob.glob(globfiles)
if len(images) < 10:
    print("Only found {} images!".format(len(images)))
    print("Not enough images! 10 or more images are required!")
    sys.exit()

# Termination criteria
# Read 3rd link of reference docs
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, attempts, epsilon)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
# TODO: If we know exactly what the size of the squares are (say 30mm) \
# then we should pass in (0,0), (30,0), (60,0) instead, which will give the result in mm instead of pixel
objp = np.zeros((num_row*num_col, 3), np.float32)
# Assume chessboard is kept stationary at XY plane, so Z = 0 always, camera was moved accordingly
objp[:,:2] = np.mgrid[0:num_row, 0:num_col].T.reshape(-1,2)

# Arrays to store object points and images points for all the images
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image planes

# Variables for final calculation
imgNotGood = []
num_pattern = 0
for fname in images:
    print("Checking {}".format(str(fname)))
    # Read image in greyscale
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (num_row, num_col), None)
    if ret:
        print('Pattern found! Now processing {}!'.format(str(fname)))
        num_pattern += 1
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        #Draw and display the corners
        cv.drawChessboardCorners(img, (num_row, num_col), corners2, ret)
        cv.imshow('Drawn Chessboard Corners', img)
        cv.waitKey(500) # Show images for 0.5s
    else:
        imgNotGood.append(fname)

cv.destroyAllWindows()

if num_pattern > 1:
    print('Found {} good images, out of {} total images'.format(num_pattern, len(images)))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Initialize mean_error for calculation
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    print("\nMean error: {}".format(mean_error/len(objpoints)))

    # Saving calibration result
    text_dump = "Mean error: {}\n\n".format(mean_error/len(objpoints))
    text_dump += 'Intrinsic Matrix:\n'
    text_dump += '{}\n\n'.format(str(mtx))
    text_dump += 'Rotation vectors:\n[\n'
    for i in rvecs:
        text_dump += '  {} {} {}\n'.format(i[0], i[1], i[2])
    text_dump += ']\n\nTranslation vectors:\n[\n'
    for i in tvecs:
        text_dump += '  {} {} {}\n'.format(i[0], i[1], i[2])
    text_dump += ']\n'

    with open('calib_result.txt', 'w') as result_file:
        result_file.write(text_dump)
    print('Calibration results saved to calib_result.txt!')
    # Stop Timer
    hours, rem = divmod((time.time() - stt), 3600)
    minutes, sec = divmod(rem, 60)
    print("Elapsed Time:\n{:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), sec))
else:
    print('Did not found enough good images for camera calibration. Try again')
    sys.exit()
