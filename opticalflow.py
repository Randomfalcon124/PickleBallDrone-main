import cv2
import numpy as np
import pyrealsense2 as rs

def compute_optical_flow(prev_img, next_img):
    """
    Computes dense optical flow using Farneback's method.
    Args:
        prev_img (np.ndarray): Previous grayscale image.
        next_img (np.ndarray): Next grayscale image.
    Returns:
        flow (np.ndarray): Optical flow vectors for each pixel.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_img, next_img, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    return flow

def draw_flow(img, flow, step=16):
    """
    Draws optical flow vectors on the image.
    Args:
        img (np.ndarray): Input image (BGR).
        flow (np.ndarray): Optical flow vectors.
        step (int): Step size for sampling flow vectors.
    Returns:
        vis (np.ndarray): Image with flow vectors drawn.
    """
    h, w = img.shape[:2]
    vis = img.copy()
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    for (x1, y1), (x2, y2) in lines:
        cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)
    return vis

def get_intrinsics(profile):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.ppx, intr.ppy
    return fx, fy, cx, cy

def flow_to_3d(flow, depth, fx, fy, cx, cy):
    """
    Converts 2D optical flow and depth to 3D flow vectors.
    Args:
        flow (np.ndarray): Optical flow (H, W, 2).
        depth (np.ndarray): Depth map (H, W) in meters.
        fx, fy, cx, cy: Camera intrinsics.
    Returns:
        flow_3d (np.ndarray): 3D flow vectors (H, W, 3).
    """
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    # Current 3D points
    z1 = depth
    x1 = (x - cx) * z1 / fx
    y1 = (y - cy) * z1 / fy

    # Next 3D points (after flow)
    x2_pix = x + flow[..., 0]
    y2_pix = y + flow[..., 1]
    x2_pix = np.clip(x2_pix, 0, w-1)
    y2_pix = np.clip(y2_pix, 0, h-1)
    z2 = depth[np.round(y2_pix).astype(int), np.round(x2_pix).astype(int)]
    x2 = (x2_pix - cx) * z2 / fx
    y2 = (y2_pix - cy) * z2 / fy

    flow_3d = np.stack([x2 - x1, y2 - y1, z2 - z1], axis=-1)
    return flow_3d

if __name__ == "__main__":
    # Configure Intel RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    fx, fy, cx, cy = get_intrinsics(profile)

    align = rs.align(rs.stream.color)

    # Get first frames
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        print("Failed to grab first frame.")
        exit(1)
    prev_color = np.asanyarray(color_frame.get_data())
    prev_gray = cv2.cvtColor(prev_color, cv2.COLOR_BGR2GRAY)
    prev_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001  # mm to meters

    # Initialize camera position (meters)
    camera_position = np.zeros(3, dtype=np.float32)

    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            break
        color = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001  # mm to meters

        flow = compute_optical_flow(prev_gray, gray)
        flow_vis = draw_flow(color, flow)

        # Compute 3D flow
        flow_3d = flow_to_3d(flow, depth, fx, fy, cx, cy)

        # Mask out invalid depth and outliers
        valid_mask = (depth > 0.2) & (depth < 5.0) & np.all(np.isfinite(flow_3d), axis=-1)
        flow_3d_valid = flow_3d[valid_mask]

        # Use median instead of mean for robustness
        if flow_3d_valid.shape[0] > 0:
            avg_3d_flow = np.nanmedian(flow_3d_valid, axis=0)
        else:
            avg_3d_flow = np.zeros(3, dtype=np.float32)

        # Integrate average 3D flow to estimate position
        camera_position += avg_3d_flow

        pos_text = (
            f"3D Flow (m/frame): dx={avg_3d_flow[0]:.3f}, dy={avg_3d_flow[1]:.3f}, dz={avg_3d_flow[2]:.3f}"
        )
        pos_text2 = (
            f"Camera Position (m): x={camera_position[0]:.3f}, y={camera_position[1]:.3f}, z={camera_position[2]:.3f}"
        )
        cv2.putText(flow_vis, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(flow_vis, pos_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow('3D Optical Flow', flow_vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

        prev_gray = gray
        prev_depth = depth

    print(f"Final Camera Position (m): x={camera_position[0]:.3f}, y={camera_position[1]:.3f}, z={camera_position[2]:.3f}")
    pipeline.stop()
    cv2.destroyAllWindows()