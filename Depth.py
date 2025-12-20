import pyrealsense2 as rs
import numpy as np
import cv2
from collections import deque
import statistics
from typing import List, Tuple

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Configure streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline_profile = pipeline.start(config)

# Get depth scale and intrinsics
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
depth_stream = pipeline_profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

try:
    while True:
        # Wait for frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Create depth visualization with multiple contour levels
        depth_viz = np.zeros_like(color_image)
        all_contours_viz = np.zeros_like(color_image)  # New visualization for all contours
        
        # Create contours at different depth levels
        max_depth = 3200  # Set max depth to 3.0m (depth is in mm)
        min_depth = 280   # Set min depth to 0.28m (depth is in mm)
        num_levels = 10
        
        closest_depth = float('inf')
        closest_center = None
        closest_3d_pos = None

        for i in range(num_levels):
            threshold = (i + 1) * max_depth / num_levels
            # Only consider depths between min_depth and max_depth
            mask = ((depth_image < threshold) & 
                   (depth_image < max_depth) & 
                   (depth_image > min_depth)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create visualization image by copying color image
            vis_image = color_image.copy()
        
            # Draw all contours in the visualization
            cv2.drawContours(all_contours_viz, contours, -1, (0, 255, 0), 1)
            
            # Filter for circular/oval contours
            for contour in contours:
                # Calculate circularity metrics
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Get contour bounds to check if it touches edges
                x, y, w, h = cv2.boundingRect(contour)
                touches_edge = (x <= 5 or y <= 5 or 
                              x + w >= depth_image.shape[1] - 5 or 
                              y + h >= depth_image.shape[0] - 5)
                
                if area > 100 and not touches_edge:  # Filter out tiny contours and edge contours
                    # Calculate circularity
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # Replace the smallest_area check with closest depth check
                    if circularity > 0.7:
                        # Draw the circular contour in the depth visualization
                        color = (0, int(255 * (1 - i/num_levels)), int(255 * i/num_levels))
                        cv2.drawContours(depth_viz, [contour], -1, color, 2)
                        
                        # Get the average depth for this contour
                        mask = np.zeros_like(depth_image)
                        cv2.drawContours(mask, [contour], -1, 255, -1)
                        mean_depth = cv2.mean(depth_image, mask=mask.astype(np.uint8))[0]
                        
                        # Get raw depth in meters (depth_scale converts from depth units to meters)
                        scaled_depth = mean_depth * depth_scale
                        
                        if scaled_depth < closest_depth and scaled_depth > (min_depth/1000) and scaled_depth < (max_depth/1000):
                            closest_depth = scaled_depth
                            M = cv2.moments(contour)
                            if M["m00"] != 0:
                                # Get center in depth image coordinates
                                center_x = int(M["m10"] / M["m00"])
                                center_y = int(M["m01"] / M["m00"])
                                closest_center = (center_x, center_y)
                                
                                # Get actual depth from depth frame for precise measurement
                                depth = depth_frame.get_distance(center_x, center_y)
                                
                                # Deproject to 3D coordinates using intrinsics
                                current_3d_pos = rs.rs2_deproject_pixel_to_point(
                                    depth_intrinsics,
                                    [center_x, center_y],
                                    depth
                                )
                                current_3d_pos = (current_3d_pos[0], -current_3d_pos[1], current_3d_pos[2])
                                closest_3d_pos = current_3d_pos

                                

        # Replace smallest_center check with closest_center
        if closest_center is not None:
            # Draw red dot on color image
            cv2.circle(color_image, closest_center, 5, (0, 0, 255), -1)
            
            # Draw red dot on depth visualization
            cv2.circle(depth_viz, closest_center, 5, (0, 0, 255), -1)
            
            # Draw red dot on contour visualization
            cv2.circle(all_contours_viz, closest_center, 5, (0, 0, 255), -1)
            
            # Add position text to all views
            pos_text = f"Pos (m): ({closest_3d_pos[0]:.2f}, {closest_3d_pos[1]:.2f}, {closest_3d_pos[2]:.2f})"
            cv2.putText(color_image, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(depth_viz, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(all_contours_viz, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Print 3D position to console
            print(f"3D Position (m): X={closest_3d_pos[0]:.3f}, Y={closest_3d_pos[1]:.3f}, Z={closest_3d_pos[2]:.3f}")

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Depth', depth_viz)
        cv2.namedWindow('All Contours', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('All Contours', all_contours_viz)
        
        if cv2.waitKey(1) == 27:  # ESC to exit
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()