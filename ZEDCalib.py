import cv2
import numpy as np
import glob
import os

# === USER-DEFINED PARAMETERS ===
square_size = 0.02  # Size of one checkerboard square in meters
checkerboard_size = (8, 5)  # Number of inner corners per chessboard row and column

# Prepare object points based on checkerboard dimensions and square size
objp = np.zeros((np.prod(checkerboard_size), 3), np.float32)
objp[:, :2] = np.indices(checkerboard_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []         # 3D real-world points
imgpoints_left = []    # 2D points in left image plane
imgpoints_right = []   # 2D points in right image plane

# Load left and right images
left_images = sorted(glob.glob("left/*.png"))
right_images = sorted(glob.glob("right/*.png"))

if len(left_images) == 0 or len(right_images) == 0:
    print("No images found in 'left/' or 'right/' folders. Please add calibration images.")
    exit(1)

print(f"Found {len(left_images)} left images and {len(right_images)} right images.")

# Loop through image pairs and find chessboard corners
for left_file, right_file in zip(left_images, right_images):
    print(f"Processing pair:\n  Left: {left_file}\n  Right: {right_file}")
    img_left = cv2.imread(left_file)
    img_right = cv2.imread(right_file)

    if img_left is None or img_right is None:
        print(f"Error loading images:\n  {left_file}\n  {right_file}")
        continue

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, checkerboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, checkerboard_size, None)

    print(f"  Left corners found: {ret_left}")
    print(f"  Right corners found: {ret_right}")

    # Visualize corners for debugging
    img_left_vis = img_left.copy()
    img_right_vis = img_right.copy()

    if ret_left:
        cv2.cornerSubPix(gray_left, corners_left, (11,11), (-1,-1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(img_left_vis, checkerboard_size, corners_left, ret_left)
        imgpoints_left.append(corners_left)
    else:
        print("  Warning: Chessboard corners NOT found in left image.")

    if ret_right:
        cv2.cornerSubPix(gray_right, corners_right, (11,11), (-1,-1),
                         criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv2.drawChessboardCorners(img_right_vis, checkerboard_size, corners_right, ret_right)
        imgpoints_right.append(corners_right)
    else:
        print("  Warning: Chessboard corners NOT found in right image.")

    # Only add object points if both corners found
    if ret_left and ret_right:
        objpoints.append(objp)
    else:
        print("  Skipping this pair since corners were not found in both images.")

    # Show images with drawn corners
    cv2.imshow('Left Image Corners', img_left_vis)
    cv2.imshow('Right Image Corners', img_right_vis)
    key = cv2.waitKey(500)  # Display for 500 ms, press any key to continue sooner
    if key == 27:  # ESC key to quit early
        print("Calibration aborted by user.")
        exit(0)

cv2.destroyAllWindows()

if len(objpoints) == 0:
    print("No valid image pairs with detected corners found. Cannot calibrate.")
    exit(1)

print(f"\nTotal valid pairs for calibration: {len(objpoints)}")

# Calibrate left and right cameras individually
ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_right.shape[::-1], None, None)

print("\nIndividual camera calibration done.")

# Stereo calibration to get rotation and translation between cameras
flags = cv2.CALIB_FIX_INTRINSIC
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_left, dist_left, mtx_right, dist_right,
    gray_left.shape[::-1],
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=flags)

print("\nStereo calibration done.")

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx_left, dist_left, mtx_right, dist_right,
    gray_left.shape[::-1], R, T, alpha=0)

# Compute undistort and rectify maps
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtx_left, dist_left, R1, P1, gray_left.shape[::-1], cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtx_right, dist_right, R2, P2, gray_right.shape[::-1], cv2.CV_16SC2)

print("\nCalibration successful!")
print(f"Focal Length (pixels): {P1[0, 0]:.2f}")
print(f"Baseline (meters): {-T[0, 0]:.3f}")
print("Q Matrix:\n", Q)

# Save calibration results
np.savez("stereo_calibration.npz",
         mtx_left=mtx_left, dist_left=dist_left,
         mtx_right=mtx_right, dist_right=dist_right,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         left_map1=left_map1, left_map2=left_map2,
         right_map1=right_map1, right_map2=right_map2)

print("\nCalibration data saved to 'stereo_calibration.npz'.")
