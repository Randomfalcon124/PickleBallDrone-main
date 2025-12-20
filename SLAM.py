import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from collections import deque

# --- 1. Camera Configuration ---
cap = cv2.VideoCapture(1)  # Change index if needed
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# --- 2. Parameters ---
SMOOTHING_WINDOW = 5
KEYFRAME_ROTATION_THRESHOLD = 0.15
KEYFRAME_TRANSLATION_THRESHOLD = 0.1

# --- 3. Feature Detection Setup ---
orb = cv2.ORB_create(nfeatures=2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# --- 4. Variables ---
prev_img = None
prev_kps = None
prev_desc = None
pose = np.eye(4)
pose_history = deque(maxlen=SMOOTHING_WINDOW)
prev_R = np.eye(3)
prev_t = np.zeros(3)
frame_count = 0
keyframes = []

# --- 5. Helper Functions ---
def filter_pose(R, t, prev_R, prev_t):
    translation_diff = np.linalg.norm(t - prev_t)
    if translation_diff > 1.0:  # Adjust as needed
        return False
    r1 = Rotation.from_matrix(prev_R)
    r2 = Rotation.from_matrix(R)
    rotation_diff = np.abs((r1.inv() * r2).magnitude())
    if rotation_diff > 1.0:  # Adjust as needed
        return False
    return True

def is_keyframe(current_R, current_t, last_keyframe):
    if not keyframes:
        return True
    last_pose = last_keyframe['pose']
    rel_R = current_R @ np.linalg.inv(last_pose[:3, :3])
    rel_t = current_t - last_pose[:3, 3]
    rotation_magnitude = np.abs(Rotation.from_matrix(rel_R).magnitude())
    translation_magnitude = np.linalg.norm(rel_t)
    return (rotation_magnitude > KEYFRAME_ROTATION_THRESHOLD or 
            translation_magnitude > KEYFRAME_TRANSLATION_THRESHOLD)

def create_keyframe(R, t, pose, frame_count, kps=None, desc=None):
    return {
        'R': R.copy(),
        't': t.copy(),
        'pose': pose.copy(),
        'frame': frame_count,
        'kps': [kp for kp in kps] if kps is not None else None,
        'desc': desc.copy() if desc is not None else None,
    }

try:
    while True:
        ret, img = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = orb.detectAndCompute(gray, None)

        if prev_kps is not None and len(kps) > 0:
            matches = bf.match(prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            if len(matches) >= 8:
                pts1 = np.float32([prev_kps[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kps[m.trainIdx].pt for m in matches])

                # Estimate Essential matrix (no depth, monocular SLAM)
                E, mask = cv2.findEssentialMat(pts1, pts2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1.0)
                if E is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, focal=1.0, pp=(0., 0.))

                    if filter_pose(R, t.flatten(), prev_R, prev_t):
                        curr_transform = np.eye(4)
                        curr_transform[:3, :3] = R
                        curr_transform[:3, 3] = t.flatten()
                        pose = pose @ np.linalg.inv(curr_transform)
                        pose_history.append(pose.copy())
                        prev_R = R.copy()
                        prev_t = t.flatten().copy()

                        cv2.rectangle(img, (10, 10), (300, 80), (255, 255, 255), -1)
                        cv2.putText(img, f"X: {pose[0,3]:.3f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                        cv2.putText(img, f"Y: {pose[1,3]:.3f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                        cv2.putText(img, f"Z: {pose[2,3]:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
                        cv2.putText(img, f"Matches: {len(matches)}", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

                        if is_keyframe(R, t.flatten(), keyframes[-1] if keyframes else None):
                            keyframes.append(create_keyframe(
                                R=R,
                                t=t.flatten(),
                                pose=pose,
                                frame_count=frame_count,
                                kps=kps,
                                desc=desc
                            ))
                        frame_count += 1

        prev_img = gray.copy()
        prev_kps = kps
        prev_desc = desc

        cv2.imshow('SLAM', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()