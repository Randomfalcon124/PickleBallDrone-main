import cv2
import numpy as np
import glob

def calibrate_camera(calib_images_path, chessboard_size=(9, 6), square_size=1.0):
    # Prepare object points (0,0,0), (1,0,0), ..., (8,5,0) * square_size
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    images = glob.glob(calib_images_path + '/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(100)

    cv2.destroyAllWindows()

    if not objpoints or not imgpoints:
        print("No corners found in images. Calibration failed.")
        return None

    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    return mtx, dist, rvecs, tvecs

if __name__ == "__main__":
    # Change the path to your calibration images folder
    calib_images_path = "calibration_images"
    calibrate_camera(calib_images_path)