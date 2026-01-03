import pyrealsense2 as rs
import numpy as np
import cv2
from typing import List, Tuple
import datetime

BAG_FILE = "20250703_224252.bag" # Customize this to your .bag file's full path
START_FRAME_NUMBER = 0 # Starting from frame 0 for simplicity

# --- Auto-Play Feature Configuration ---
# Set this to an integer frame number to automatically play until that frame is reached.
# Set to None to disable auto-play and start in a paused state.
AUTO_PLAY_TARGET_FRAME = 1200 # Example: Auto-play until depth frame 200
# ---------------------------------------------

# --- Trajectory Prediction Configuration (Moved to trajectory_plotter_code) ---
# These constants are now defined in trajectory_plotter_code.py
# GRAVITY_ACCELERATION_Z = -9.81 # m/s^2, assuming Z is 'up' in the remapped plot coordinates
# TRAJECTORY_PREDICTION_STEPS = 50 # Number of points to generate for the trajectory line
# TRAJECTORY_FIT_DEGREE = 1 # Degree of polynomial to fit to X, Y, Z coordinates for trajectory estimation.
                          # 1 for linear (constant velocity), 2 for quadratic (constant acceleration in X, Y).
                          # Minimum (degree + 1) points required for fitting.
# ------------------------------------------------

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable device from file (bag file)
config.enable_device_from_file(BAG_FILE, repeat_playback=True)

# Start streaming
pipeline_profile = pipeline.start(config)

# If using a bag file, set playback to non-real-time to process as fast as possible
device = pipeline_profile.get_device()
playback = None
if device.as_playback():
    playback = device.as_playback()
    playback.set_real_time(False) # CRUCIAL: Disable real-time playback for frame-by-frame stepping

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Get depth scale and intrinsics
try:
    depth_sensor = pipeline_profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
except RuntimeError:
    print("WARNING: Depth sensor not available in the stream profile.")
    depth_scale = 0.001 # Default to a common depth scale if not found

try:
    color_stream = pipeline_profile.get_stream(rs.stream.color)
    aligned_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
except RuntimeError:
    print("WARNING: Color stream not available in the stream profile. Intrinsics will not be set.")
    aligned_intrinsics = None # Set to None if color stream is not available

# --- Initialize RealSense Post-Processing Filters ---
spatial_filter = rs.spatial_filter()
spatial_filter.set_option(rs.option.filter_magnitude, 2) # Strength of the filter (1-5)
spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5) # Alpha factor (0-1)
spatial_filter.set_option(rs.option.holes_fill, 1) # How holes are filled: 0=None, 1=FARTHEST_FROM_AROUND, 2=NEAREST_FROM_AROUND

# --- Distance Parameters ---
MAX_DISPLAY_DISTANCE = 10000.0
MAX_DETECTION_DISTANCE = 10.0 # Max detection distance for plotting and real-time display
COLORMAP_ALPHA = 255.0 / (MAX_DISPLAY_DISTANCE * 1000)

# --- Colorfulness Masking Threshold ---
SATURATION_THRESHOLD = 100

# --- Brightest Object Detection Parameters ---
BRIGHTNESS_THRESHOLD = 75
DETECTION_CIRCLE_RADIUS = 5 # Radius for the detection circle drawn and for depth sampling

# --- Modified FUNCTION: Create a Colorbar with Labels (fixed 0 to max_meters) ---
def create_colorbar_with_labels(height, width, max_meters, colormap_type=cv2.COLORMAP_JET, num_labels=5):
    colorbar = np.zeros((height, width, 3), dtype=np.uint8)
    gradient_8bit = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    gradient_colored = cv2.applyColorMap(gradient_8bit, colormap_type)

    for i in range(height):
        colorbar[i, :] = gradient_colored[height - 1 - i, 0]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (255, 255, 255) # White color for text

    if max_meters <= 0.01:
        cv2.putText(colorbar, "0.0m", (5, height - 10), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(colorbar, "MAX", (5, 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return colorbar

    for i in range(num_labels):
        label_meter_val = (max_meters / (num_labels - 1)) * i
        normalized_pos = label_meter_val / max_meters
        y_pos = int((1 - normalized_pos) * (height - 1))

        if y_pos < 10: y_pos = 10
        if y_pos > height - 10: y_pos = height - 10

        text = f"{label_meter_val:.1f}m"
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = width - text_size[0] - 5
        text_y = y_pos + text_size[1] // 2

        cv2.putText(colorbar, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    return colorbar

colorbar_img = create_colorbar_with_labels(
    height=480,
    width=60,
    max_meters=MAX_DISPLAY_DISTANCE
)

# --- Playback State Variables ---
paused = True
current_frame_number = 0 # This will now be updated directly from depth_frame.get_frame_number()
total_frames = 0
fps = 30

# --- Auto-play state initialization ---
_auto_playing = False
if playback and AUTO_PLAY_TARGET_FRAME is not None:
    if AUTO_PLAY_TARGET_FRAME < START_FRAME_NUMBER:
        print(f"WARNING: AUTO_PLAY_TARGET_FRAME ({AUTO_PLAY_TARGET_FRAME}) is less than START_FRAME_NUMBER ({START_FRAME_NUMBER}). Auto-play will not reach target in forward playback.")
    else:
        paused = False # Start playing automatically
        _auto_playing = True
        print(f"Initiating auto-play to depth frame: {AUTO_PLAY_TARGET_FRAME}")
# ---------------------------------------------

# --- Matplotlib 3D Plot Initialization ---
# detected_points_3d will now store tuples of (frame_number, x, y, z, timestamp_ms)
detected_points_3d: List[Tuple[int, float, float, float, float]] = []
# -----------------------------------------


if playback:
    try:
        color_profile = color_stream.as_video_stream_profile()
        fps = color_profile.fps()

        if fps <= 0:
            raise ValueError(f"Invalid FPS obtained from stream: {fps}. Cannot calculate total frames.")

        duration_in_seconds = playback.get_duration() / 1e9
        total_frames = int(duration_in_seconds * fps)

        target_time_seconds = START_FRAME_NUMBER * (1.0 / fps)
        playback.seek(datetime.timedelta(seconds=target_time_seconds))

    except Exception as e:
        print(f"WARNING: An error occurred during playback initialization/seeking: {e}")
        playback = None
        total_frames = 0
        print("Playback functionality will be disabled due to error in initialization.")
else:
    print("Not running from a bag file, playback controls will not work.")


try:
    while True:
        if playback:
            current_playback_time_sec = playback.get_position() / 1e9
            temp_frame_for_title = int(current_playback_time_sec * fps) if fps > 0 else 0

            # --- Update window title based on auto-play state ---
            if _auto_playing:
                cv2.setWindowTitle('Frame Viewer', f"AUTO-PLAYING to {AUTO_PLAY_TARGET_FRAME}! Frame: {temp_frame_for_title} / {total_frames if total_frames > 0 else '??'}")
            else:
                cv2.setWindowTitle('Frame Viewer', f"Frame: {temp_frame_for_title} / {total_frames if total_frames > 0 else '??'} (Press SPACE to {'Play' if paused else 'Pause'})")
            # -----------------------------------------------------------
        else:
            cv2.setWindowTitle('Frame Viewer', "Live Stream (No Playback Controls)")

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            key = cv2.waitKey(1)
            if _auto_playing and key != -1: # If a key is pressed during auto-play
                _auto_playing = False
                paused = True
            if key == 27: break
            continue
        else:
            # --- IMPORTANT CHANGE: Update current_frame_number from the actual depth frame ---
            current_frame_number = depth_frame.get_frame_number()
            # Get the timestamp of the current depth frame (in milliseconds)
            current_timestamp_ms = depth_frame.get_timestamp()
            # ---------------------------------------------------------------------------------

            # --- Check for auto-play completion ---
            if _auto_playing:
                if current_frame_number >= AUTO_PLAY_TARGET_FRAME:
                    _auto_playing = False
                    paused = True
                    print(f"Auto-play reached target frame {AUTO_PLAY_TARGET_FRAME}. Pausing.")
            # ------------------------------------------

            filtered_depth_frame = spatial_filter.process(depth_frame)

            depth_image = np.asanyarray(filtered_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # --- Re-added: Ensure depth_image matches color_image dimensions for broadcasting ---
            if depth_image.shape[:2] != color_image.shape[:2]:
                depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            # -----------------------------------------------------------------------------

            depth_image_raw = np.asanyarray(depth_frame.get_data())
            # Re-added: Also resize raw depth image if it's used for display and needs to match color
            if depth_image_raw.shape[:2] != color_image.shape[:2]:
                depth_image_raw = cv2.resize(depth_image_raw, (color_image.shape[1], color_image.shape[0]), interpolation=cv2.INTER_NEAREST)


            depth_colormap_raw = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image_raw, alpha=COLORMAP_ALPHA),
                cv2.COLORMAP_JET
            )

            hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            saturation_channel = hsv_image[:,:,1]
            value_channel = hsv_image[:,:,2] # Get the Value (brightness) channel

            color_filter_mask_bool = (saturation_channel > SATURATION_THRESHOLD)

            depth_in_meters = depth_image.astype(np.float32) * depth_scale

            distance_filter_mask = (depth_in_meters <= MAX_DISPLAY_DISTANCE)

            # --- Generate Contours specifically for the Depth Stream (based on distance only) ---
            depth_only_binary_mask = (distance_filter_mask * 255).astype(np.uint8)
            depth_contours, _ = cv2.findContours(depth_only_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # ---------------------------------------------------------------------------------------

            # Combined mask for what's displayed in "Filtered Color" (this remains as before)
            combined_mask = distance_filter_mask & color_filter_mask_bool

            filtered_color_for_display = np.copy(color_image)
            filtered_color_for_display[~combined_mask] = 0

            # Contours for the Filtered Color stream (based on combined mask)
            gray_for_filtered_contours = cv2.cvtColor(filtered_color_for_display, cv2.COLOR_BGR2GRAY)
            _, non_black_mask = cv2.threshold(gray_for_filtered_contours, 1, 255, cv2.THRESH_BINARY)
            contours_on_filtered, _ = cv2.findContours(non_black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(filtered_color_for_display, contours_on_filtered, -1, (255, 0, 0), 1) # Blue contours

            brightest_blob_center = None
            brightest_blob_3d_pos = None
            
            # Initialize brightest_pixel_u and brightest_pixel_v for drawing
            brightest_pixel_u, brightest_pixel_v = -1, -1

            gray_filtered_color_for_brightest = cv2.cvtColor(filtered_color_for_display, cv2.COLOR_BGR2GRAY)

            _, bright_mask = cv2.threshold(gray_filtered_color_for_brightest, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)

            bright_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            max_mean_brightness_value = -1
            best_bright_contour = None

            if bright_contours:
                for i, contour in enumerate(bright_contours):
                    area = cv2.contourArea(contour)

                    contour_mask = np.zeros_like(gray_filtered_color_for_brightest, dtype=np.uint8)
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                    valid_pixels_in_contour = gray_filtered_color_for_brightest[contour_mask > 0]

                    if valid_pixels_in_contour.size > 0:
                        current_mean_brightness = np.mean(valid_pixels_in_contour)

                        if current_mean_brightness > max_mean_brightness_value:
                            max_mean_brightness_value = current_mean_brightness
                            best_bright_contour = contour

                if best_bright_contour is not None:
                    # --- NEW LOGIC: Draw border of pixels similar to the brightest pixel in both brightness and saturation ---
                    contour_pixel_mask = np.zeros_like(saturation_channel, dtype=np.uint8)
                    cv2.drawContours(contour_pixel_mask, [best_bright_contour], -1, 255, -1)

                    ys, xs = np.where(contour_pixel_mask == 255)
                    brightness_vals = value_channel[ys, xs]
                    saturation_vals = saturation_channel[ys, xs]

                    # Find the brightest pixel in the contour
                    max_brightness = np.max(brightness_vals)
                    max_idx = np.argmax(brightness_vals)
                    brightest_pixel_u = xs[max_idx]
                    brightest_pixel_v = ys[max_idx]
                    brightest_saturation = saturation_vals[max_idx]

                    # Thresholds for similarity (tune as needed)
                    BRIGHTNESS_SIM_THRESH = 1000
                    SATURATION_SIM_THRESH = 1000

                    # Keep only pixels close to the brightest pixel in both brightness and saturation
                    similar_mask = (
                        (np.abs(brightness_vals - max_brightness) < BRIGHTNESS_SIM_THRESH) &
                        (np.abs(saturation_vals - brightest_saturation) < SATURATION_SIM_THRESH)
                    )
                    inlier_xs = xs[similar_mask]
                    inlier_ys = ys[similar_mask]

                    # Create a mask for inlier pixels
                    inlier_mask = np.zeros_like(contour_pixel_mask, dtype=np.uint8)
                    inlier_mask[inlier_ys, inlier_xs] = 255

                    # Find the border of the inlier region
                    inlier_contours, _ = cv2.findContours(inlier_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(filtered_color_for_display, inlier_contours, -1, (0, 255, 255), 2)  # Yellow border

                    # Use centroid of similar pixels as the ball position (if any)
                    if len(inlier_xs) > 0:
                        centroid_u = int(np.mean(inlier_xs))
                        centroid_v = int(np.mean(inlier_ys))
                        brightest_pixel_u = centroid_u
                        brightest_pixel_v = centroid_v
                    # --- END NEW LOGIC ---

                    # --- Depth sampling at the selected pixel (brightest_pixel_u, brightest_pixel_v) ---
                    depth_at_brightest_pixel_meters = 0.0
                    if brightest_pixel_u != -1 and brightest_pixel_v != -1:
                        image_height, image_width = depth_image.shape[0], depth_image.shape[1]
                        
                        # Ensure pixel coordinates are within image bounds before accessing
                        if 0 <= brightest_pixel_v < image_height and 0 <= brightest_pixel_u < image_width:
                            raw_depth_at_brightest_pixel = depth_image[brightest_pixel_v, brightest_pixel_u]
                            if raw_depth_at_brightest_pixel > 0: # Check for valid depth (0 means no data)
                                depth_at_brightest_pixel_meters = raw_depth_at_brightest_pixel * depth_scale
                        
                    # --- END Depth sampling ---

                    M = cv2.moments(best_bright_contour)
                    if M["m00"] != 0:
                        brightest_blob_center_x = int(M["m10"] / M["m00"])
                        brightest_blob_center_y = int(M["m01"] / M["m00"])
                        brightest_blob_center = (brightest_blob_center_x, brightest_blob_center_y)

                        # --- Step 4: Deproject using the selected pixel and its depth ---
                        # Always attempt to deproject if conditions allow, regardless of MAX_DETECTION_DISTANCE
                        if brightest_pixel_u != -1 and brightest_pixel_v != -1 and depth_at_brightest_pixel_meters > 0 and aligned_intrinsics is not None:
                            brightest_blob_3d_pos_raw = rs.rs2_deproject_pixel_to_point(
                                aligned_intrinsics,
                                [brightest_pixel_u, brightest_pixel_v], # Use selected pixel for deprojection
                                depth_at_brightest_pixel_meters # Use depth at selected pixel
                            )
                            brightest_blob_3d_pos = (brightest_blob_3d_pos_raw[0], -brightest_blob_3d_pos_raw[1], brightest_blob_3d_pos_raw[2])
                            
                            # --- Store frame number and timestamp with 3D position ---
                            # Store all valid 3D positions regardless of AUTO_PLAY_TARGET_FRAME for later filtering
                            detected_points_3d.append((current_frame_number, brightest_blob_3d_pos[0], brightest_blob_3d_pos[1], brightest_blob_3d_pos[2], current_timestamp_ms))
                            # ------------------------------------
                        else:
                            brightest_blob_3d_pos = None # Keep as None if 3D position cannot be calculated at all

            # --- Overlay brightest blob info on filtered_color_for_display ---
            # The circle is always drawn at brightest_blob_center (centroid) for display,
            # but depth is sampled from the brightest pixel as per new logic.
            if brightest_blob_center is not None:
                # Determine display color based on MAX_DETECTION_DISTANCE for real-time feedback
                if brightest_blob_3d_pos is not None and brightest_blob_3d_pos[2] <= MAX_DETECTION_DISTANCE:
                    circle_color = (0, 255, 255) # Yellow for color image (within range)
                    text_color = (0, 255, 255)
                    depth_circle_color = (0, 255, 0) # Green for depth map (within range)
                    depth_text_color = (0, 0, 0) # Black for depth map
                    overlay_text = (f"Pos(m): ({brightest_blob_3d_pos[0]:.2f}, "
                                    f"{brightest_blob_3d_pos[1]:0.2f}, "
                                    f"{brightest_blob_3d_pos[2]:.2f})")
                elif brightest_blob_3d_pos is not None and brightest_blob_3d_pos[2] > MAX_DETECTION_DISTANCE:
                    circle_color = (0, 165, 255) # Orange for color image (out of range)
                    text_color = (0, 165, 255)
                    depth_circle_color = (0, 165, 255) # Orange for depth map (out of range)
                    depth_text_color = (0, 0, 0)
                    overlay_text = (f"OUT OF RANGE({brightest_blob_3d_pos[2]:.2f}m)")
                else:
                    overlay_text = "Detection Discarded" # Cannot calculate 3D position at all
                    circle_color = (0, 0, 255) # Red
                    text_color = (0, 0, 255)
                    depth_circle_color = (0, 0, 255) # Red
                    depth_text_color = (0, 0, 0)

                # cv2.circle(filtered_color_for_display, brightest_blob_center, DETECTION_CIRCLE_RADIUS, circle_color, 2)  # <-- REMOVE THIS LINE
                cv2.putText(filtered_color_for_display, overlay_text,
                            (brightest_blob_center[0] + 15, brightest_blob_center[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)

                # --- Visualize the exact depth sampling pixel on the color stream ---
                # Use a distinct color, e.g., cyan (255, 255, 0)
                if brightest_pixel_u != -1 and brightest_pixel_v != -1: # Use the brightest pixel for drawing
                    cv2.circle(color_image, (brightest_pixel_u, brightest_pixel_v), 1, (255, 255, 0), -1) # Filled cyan circle on raw color
                    cv2.circle(filtered_color_for_display, (brightest_pixel_u, brightest_pixel_v), 1, (255, 255, 0), -1) # Filled cyan circle on filtered color
                    cv2.putText(color_image, "Depth Pixel", (brightest_pixel_u + 10, brightest_pixel_v - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)


                # Make a copy of raw depth colormap for drawing contours and blob info
                depth_display_with_contours = np.copy(depth_colormap_raw)

                # Overlay DEPTH CONTOURS on the raw depth stream (using black color)
                cv2.drawContours(depth_display_with_contours, depth_contours, -1, (0, 0, 0), 1) # Black depth contours

                # Stack the depth colormap with the colorbar
                depth_display_with_colorbar = np.hstack((depth_display_with_contours, colorbar_img))

                # Draw the brightest blob center on the depth stream (using centroid for drawing)
                cv2.circle(depth_display_with_colorbar, brightest_blob_center, DETECTION_CIRCLE_RADIUS, depth_circle_color, 2)
                cv2.putText(depth_display_with_colorbar, overlay_text,
                            (brightest_blob_center[0] + 15, brightest_blob_center[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, depth_text_color, 2, cv2.LINE_AA)

                cv2.namedWindow('Raw Depth Stream + Colorbar', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Raw Depth Stream + Colorbar', depth_display_with_colorbar)
            else:
                # If no blob center was found, just show the raw depth colormap with black contours and colorbar
                depth_display_with_contours = np.copy(depth_colormap_raw)
                # Overlay DEPTH CONTOURS even if no bright blob is found
                cv2.drawContours(depth_display_with_contours, depth_contours, -1, (0, 0, 0), 1) # Black depth contours
                depth_display_with_colorbar = np.hstack((depth_display_with_contours, colorbar_img))
                cv2.namedWindow('Raw Depth Stream + Colorbar', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Raw Depth Stream + Colorbar', depth_display_with_colorbar)


            cv2.namedWindow('Filtered Color', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Filtered Color', filtered_color_for_display)

            cv2.namedWindow('Color', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Color', color_image)

        # --- Determine waitKey duration based on current mode ---
        if _auto_playing:
            key = cv2.waitKey(1) # Play continuously during auto-play
        elif not paused:
            key = cv2.waitKey(1) # Play continuously if unpaused by user
        else:
            key = cv2.waitKey(0) # Wait indefinitely if paused by user
        # -----------------------------------------------------------

        if key == 27: # ESC key
            print("ESC pressed. Exiting.")
            break
        elif key == 32: # SPACE key
            paused = not paused
            _auto_playing = False # If space is pressed, turn off auto-play
            print(f"Playback {'PAUSED' if paused else 'PLAYING'}")
        # Right arrow key - common codes for Windows (2555904) and Linux/macOS (65363)
        elif key == 2555904 or key == 65363:
            if playback:
                old_frame_number = current_frame_number
                current_frame_number += 1
                if total_frames > 0 and current_frame_number >= total_frames:
                    print("Reached end of video.")
                    current_frame_number = total_frames - 1

                if fps > 0:
                    target_time_seconds = current_frame_number * (1.0 / fps)
                    playback.seek(datetime.timedelta(seconds=target_time_seconds))

                print(f"Seeking forward to frame {current_frame_number}")
                paused = True # Pause after manual seek
                _auto_playing = False # Ensure auto-play is off
            else:
                print("Cannot seek in live stream. Playback controls only available for bag files.")
        # Left arrow key functionality is completely removed as requested.

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

    # --- Prepare points for plotting ---
    print(f"\nTotal points collected before filtering: {len(detected_points_3d)}")
    
    plottable_points_with_timestamps = [] # Stores (remapped_x, remapped_y, remapped_z, original_timestamp_ms)
    
    for frame_num, x, y, z, timestamp_ms in detected_points_3d:
        # Filter by AUTO_PLAY_TARGET_FRAME AND MAX_DETECTION_DISTANCE
        if (AUTO_PLAY_TARGET_FRAME is None or frame_num >= AUTO_PLAY_TARGET_FRAME) and (z <= MAX_DETECTION_DISTANCE):
            # Remap coordinates for plotting as per user's request:
            # Plot X: -x (X points left)
            # Plot Y: -z (Y points towards user, since z is depth away)
            # Plot Z: y (Z points up, since y is already Y up)
            plottable_points_with_timestamps.append((-x, -z, y, timestamp_ms))
            
    print(f"Total points to be used for plotting/analysis after filtering: {len(plottable_points_with_timestamps)}")

    # Sort points by timestamp to ensure temporal order for trajectory prediction
    plottable_points_with_timestamps.sort(key=lambda p: p[3])

    # --- Print all collected points ---
    print("\n--- Collected Plottable Points (X, Y, Z, Timestamp_ms) ---")
    for p in plottable_points_with_timestamps:
        print(f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}, {p[3]:.0f})")

    # --- Final Duration Calculations ---
    if plottable_points_with_timestamps:
        min_timestamp_plotted = min(p[3] for p in plottable_points_with_timestamps)
        max_timestamp_plotted = max(p[3] for p in plottable_points_with_timestamps)
        
        total_data_duration_seconds = (max_timestamp_plotted - min_timestamp_plotted) / 1000.0
        print(f"\nTotal duration of all plotted points: {total_data_duration_seconds:.9f} seconds (from timestamp {min_timestamp_plotted} to {max_timestamp_plotted} milliseconds)")
    else:
        print("\nNo timestamps available in points_to_plot_3d to calculate total data duration.")
