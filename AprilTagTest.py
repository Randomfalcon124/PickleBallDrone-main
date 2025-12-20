import cv2
import numpy as np
from pupil_apriltags import Detector
import time  # <-- Add this import

# Initialize USB camera (change 0 to your camera index if needed)
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Disable autofocus (if supported)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 90)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 for manual mode on some cameras
cap.set(cv2.CAP_PROP_EXPOSURE, 700)

camera_matrix = np.array([
    [370.073,   0.,         369.735],
    [  0.,         444.601, 260.813],
    [  0.,           0.,           1.        ]
])


dist_coeffs = np.array([0.00572082, -0.00537963, -0.00265946, -0.00365685, -0.00619617])

# Extract fx, fy, cx, cy for AprilTag pose estimation
fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

# Specify the tag size in meters (adjust to your tag's actual size)
tag_size = 0.146  # 5 cm

# Initialize AprilTag detector
detector = Detector(families='tag25h9',
                   nthreads=16,
                   quad_decimate=3.0,
                   quad_sigma=0.0,
                   refine_edges=1,
                   decode_sharpening=0.25,
                   debug=0)

try:
    prev_time = time.time()  # <-- Initialize previous time
    while True:
        ret, color_image = cap.read()
        if not ret:
            continue

        # Undistort the image
        undistorted = cv2.undistort(color_image, camera_matrix, dist_coeffs)

        # Convert to grayscale
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        
        # Detect AprilTags with pose estimation
        results = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[fx, fy, cx, cy],
            tag_size=tag_size
        )
        
        # Draw detection results
        for r in results:
            tag_id = r.tag_id
            center = tuple(int(c) for c in r.center)
            corners = r.corners.astype(int)

            # Draw tag outline
            cv2.polylines(undistorted, [corners], True, (0, 255, 0), 2)

            # Draw tag center
            cv2.circle(undistorted, center, 5, (0, 0, 255), -1)

            # Add tag ID text
            cv2.putText(undistorted, f"ID: {tag_id}", 
                       (center[0] - 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # If pose is estimated, print translation (position)
            if r.pose_t is not None:
                t = r.pose_t  # 3x1 translation vector (in meters)
                print(f"Tag {tag_id} position (x, y, z): {t.ravel()}")

        # --- FPS calculation and display ---
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        print(f"FPS: {fps:.2f}")  # Print FPS to console
        # cv2.putText(undistorted, f"FPS: {fps:.2f}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        # -----------------------------------

        # Display the result
        cv2.imshow('AprilTag Detection', undistorted)
        
        # Print camera resolution and actual FPS
        print("Resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("FPS (actual):", cap.get(cv2.CAP_PROP_FPS))

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()