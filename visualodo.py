import open3d as o3d
import numpy as np

# Convert your color and depth frames to Open3D images
color_o3d = o3d.geometry.Image(color)
depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))  # Open3D expects depth in mm

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d, depth_o3d, convert_rgb_to_intensity=False
)

# Camera intrinsics
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=color.shape[1], height=color.shape[0], fx=fx, fy=fy, cx=cx, cy=cy
)

# Use o3d.pipelines.odometry.compute_rgbd_odometry for pose estimation