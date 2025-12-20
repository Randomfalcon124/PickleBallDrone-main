import cv2
import numpy as np

# Load calibration data
data = np.load('stereo_calibration.npz')
left_map1 = data['left_map1']
left_map2 = data['left_map2']
right_map1 = data['right_map1']
right_map2 = data['right_map2']
Q = data['Q']

# StereoSGBM parameters (start smaller and adjustable)
window_size = 3
min_disp = 0
num_disp = 16 * 6  # must be divisible by 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Read one frame to check resolution
ret, frame = cap.read()
if not ret:
    print("Cannot read frame from camera")
    exit()

frame_height, frame_width = frame.shape[:2]
single_width = frame_width // 2

print(f"Camera frame size: {frame_width}x{frame_height}")
print(f"Calibration map size: {left_map1.shape[1]}x{left_map1.shape[0]} (width x height)")

# If frame size does NOT match calibration map size, resize frame here:
if (frame_height, single_width) != (left_map1.shape[0], left_map1.shape[1]):
    print("WARNING: Frame size does not match calibration size. Resizing frames to calibration size.")
    resize_to = (left_map1.shape[1] * 2, left_map1.shape[0])  # width,height for full stereo frame

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame if needed
    if (frame_height, single_width) != (left_map1.shape[0], left_map1.shape[1]):
        frame = cv2.resize(frame, resize_to)

    # Split left/right images
    img_left = frame[:, :left_map1.shape[1]]
    img_right = frame[:, left_map1.shape[1]:left_map1.shape[1]*2]

    # Rectify
    rect_left = cv2.remap(img_left, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)

    # Debug: show rectified images
    cv2.imshow('Rectified Left', rect_left)
    cv2.imshow('Rectified Right', rect_right)

    # Grayscale
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # Debug disparity stats
    print(f"Disparity min: {disparity.min()}, max: {disparity.max()}, mean: {disparity.mean()}")

    # Normalize disparity for display (clip to valid range)
    disp_vis = (disparity - min_disp) / num_disp
    disp_vis = np.clip(disp_vis, 0, 1)

    cv2.imshow('Depth Map (Disparity)', disp_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
