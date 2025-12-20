import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

# --- Configuration for Trajectory Plotting ---
# You will paste the collected points from the output of realsense_viewer_code here.
# Example format:
# plottable_points_with_timestamps = [
#     (-0.1234, -0.5678, 0.9123, 1678886400000),
#     (-0.1250, -0.5700, 0.9150, 1678886400033),
#     # ... more points
# ]
# Original raw data - All points are restored here as requested.
raw_plottable_points_with_timestamps: List[Tuple[float, float, float, float]] = [
    (-2.3642, -9.0580, 2.2182, 1751600578947),
    (-2.0126, -7.8080, 1.9755, 1751600578980),
    (-2.0360, -7.8990, 2.0114, 1751600579013),
    (-2.3662, -9.1800, 2.3824, 1751600579047),
    (-2.0681, -8.2850, 2.1097, 1751600579080),
    (-2.2168, -8.9390, 2.2762, 1751600579113),
    (-1.8964, -7.5480, 1.9220, 1751600579147),
    (-1.8719, -7.5480, 1.8852, 1751600579180),
    (-2.1325, -8.5990, 2.1896, 1751600579214),
    (-2.0187, -8.0870, 1.9409, 1751600579214),
    (-1.8473, -7.3050, 1.7295, 1751600579280),
    (-1.8009, -7.0760, 1.6293, 1751600579314),
    (-1.7446, -6.7260, 1.4503, 1751600579347),
    (-1.6315, -6.2900, 1.2949, 1751600579380),
    (-1.5418, -5.9070, 1.1104, 1751600579414),
    (-1.5666, -5.8560, 0.9866, 1751600579447),
    (-1.4539, -5.4350, 0.7831, 1751600579480),
    (-1.4568, -5.4790, 0.6647, 1751600579514),
    (-1.4173, -5.2660, 0.5533, 1751600579547),
    (-1.9735, -7.0760, 0.4904, 1751600579580),
    (-1.4131, -4.9230, 0.2531, 1751600579614),
    (-1.2012, -4.0920, -0.0823, 1751600579680),
    (-1.2274, -3.3320, -1.1829, 1751600579847)
]

# All original points are now used for plotting and initial kinematics.
plottable_points_with_timestamps = raw_plottable_points_with_timestamps


# --- Trajectory Prediction Configuration ---
INITIAL_GRAVITY_ACCELERATION_Z = -9.81 # m/s^2, assuming Z is 'up' in the remapped plot coordinates.
                                       # This value is used directly in the physics simulation.
TRAJECTORY_PREDICTION_STEPS = 50 # Number of points to generate for the trajectory line
# ------------------------------------------------------------------------------------------

# --- Configuration for Initial Kinematics Calculation Window ---
# Number of initial points to use for fitting the polynomial and determining initial conditions.
# For X/Y velocity (linear fit), need at least 2 points.
# For Z position/velocity (linear fit for Z_adjusted), need at least 2 points.
NUM_POINTS_FOR_INITIAL_KINEMATICS = 12 
# ---------------------------------------------------------------

# --- Configuration for Weighted Regression ---
# Number of most recent points within the calculation window to apply extra weight to.
NUM_RECENT_POINTS_FOR_EXTRA_WEIGHT = 3

# Multiplier for the extra weight. E.g., 10.0 means these points are weighted 10 times more.
EXTRA_WEIGHT_MULTIPLIER = 100.0 
# ---------------------------------------------

# --- Pickleball Physical Parameters (Approximate Values) ---
PICKLEBALL_MASS_KG = 0.026 # Average mass of a pickleball (approx 26 grams)
PICKLEBALL_RADIUS_M = 0.0365 # Average radius of a pickleball (approx 36.5 mm)
PICKLEBALL_CROSS_SECTIONAL_AREA_M2 = np.pi * (PICKLEBALL_RADIUS_M**2)
AIR_DENSITY_KG_M3 = 1.225 # Standard air density at sea level

# Separate drag coefficients for horizontal and vertical movement
INITIAL_DRAG_COEFFICIENT_HORIZONTAL = 1 # Drag affecting X and Y movement
INITIAL_DRAG_COEFFICIENT_VERTICAL = 4.0 # Drag affecting Z movement (increased for slower fall)

# Note: Magnus effect (spin) is not explicitly modeled here due to lack of spin data.
# Its effect is complex (depends on spin rate and axis) and would require additional input.
# -----------------------------------------------------------

# --- Trajectory Prediction Function ---
def predict_trajectory(points_data_for_fit: List[Tuple[float, float, float, float]],
                       gravity_z: float,
                       prediction_total_duration_s: float,
                       num_steps: int,
                       num_recent_weighted: int,
                       extra_weight_mult: float,
                       drag_coefficient_horizontal: float, 
                       drag_coefficient_vertical: float,   
                       air_density: float) -> List[Tuple[float, float, float]]: 
    """
    Predicts a trajectory based on the provided historical points using physics-based simulation,
    calculating initial velocity and acceleration for each axis, and accounting for drag.
    Applies weighting to more recent points during the initial condition fitting.

    Args:
        points_data_for_fit: List of (x, y, z, timestamp_ms) for historical points used for fitting.
        gravity_z: Acceleration due to gravity along the Z-axis (e.g., -9.81 m/s^2).
        prediction_total_duration_s: The total duration (in seconds) for which to predict the trajectory.
                                     This duration starts from the last point in points_data_for_fit.
        num_steps: Number of points to generate for the trajectory line.
        num_recent_weighted: Number of most recent points in points_data_for_fit to apply extra weight to.
        extra_weight_mult: Multiplier for the extra weight.
        drag_coefficient_horizontal: The drag coefficient for horizontal (X, Y) movement.
        drag_coefficient_vertical: The drag coefficient for vertical (Z) movement.
        air_density: The air density to use in the simulation.

    Returns:
        A list of (x, y, z) tuples representing the predicted trajectory.
    """
    # Ensure enough points for a linear fit (degree 1) for all components
    if len(points_data_for_fit) < 2:
        print(f"DEBUG: Not enough points ({len(points_data_for_fit)}) for trajectory prediction (need at least 2 for linear fit). Returning only initial point.")
        if points_data_for_fit:
            return [tuple(points_data_for_fit[-1][:3])] # Return last point if only one
        return []

    # Extract relative timestamps (in seconds) for the fitting window
    t_values_ms = [p[3] for p in points_data_for_fit]
    t_origin_ms = t_values_ms[0]
    relative_t_s = [(t - t_origin_ms) / 1000.0 for t in t_values_ms]

    # Time at which to calculate initial conditions (last point used for fitting)
    t_initial_s = relative_t_s[-1] 

    # Extract remapped coordinates
    x_values = [p[0] for p in points_data_for_fit]
    y_values = [p[1] for p in points_data_for_fit]
    z_values = [p[2] for p in points_data_for_fit]

    # --- Generate Weights for Polynomial Fitting ---
    weights = np.ones(len(points_data_for_fit))
    for j in range(min(num_recent_weighted, len(points_data_for_fit))):
        weights[len(points_data_for_fit) - 1 - j] *= extra_weight_mult

    # --- Calculate Initial Position from the last point in the fitting window ---
    initial_x_pos = x_values[-1]
    initial_y_pos = y_values[-1]
    initial_z_pos = z_values[-1]

    # --- Calculate Initial Velocities (Average over fitting window, weighted) ---
    # Calculate time difference for the entire fitting window
    time_diff_for_fit_s = relative_t_s[-1] - relative_t_s[0]

    initial_x_vel = 0.0
    initial_y_vel = 0.0
    initial_z_vel_nongrav = 0.0

    if time_diff_for_fit_s > 1e-9: # Avoid division by zero
        # Use simple average velocity over the weighted window for robustness
        initial_x_vel = (x_values[-1] - x_values[0]) / time_diff_for_fit_s
        initial_y_vel = (y_values[-1] - y_values[0]) / time_diff_for_fit_s
        
        # For Z, calculate non-gravitational velocity component
        z_adjusted_first = z_values[0] - 0.5 * gravity_z * relative_t_s[0]**2
        z_adjusted_last = z_values[-1] - 0.5 * gravity_z * relative_t_s[-1]**2
        initial_z_vel_nongrav = (z_adjusted_last - z_adjusted_first) / time_diff_for_fit_s

    initial_z_vel = initial_z_vel_nongrav + gravity_z * relative_t_s[-1] # Add gravitational component

    # Non-gravitational accelerations are explicitly zero for this model
    x_accel_nongrav = 0.0
    y_accel_nongrav = 0.0
    z_accel_nongrav = 0.0
    
    # The total Z acceleration used in simulation is just gravity
    total_z_accel_for_print = gravity_z 

    print(f"\n--- Prediction Parameters (using {len(points_data_for_fit)} weighted points) ---")
    print(f"Time Origin (ms): {t_origin_ms}")
    print(f"Prediction Start Time (relative s): {t_initial_s:.4f}")
    print(f"Prediction Total Duration (s): {prediction_total_duration_s:.4f}")
    print(f"Initial Position (X, Y, Z): ({initial_x_pos:.4f}, {initial_y_pos:.4f}, {initial_z_pos:.4f})")
    print(f"Initial Velocity (X, Y, Z): ({initial_x_vel:.4f}, {initial_y_vel:.4f}, {initial_z_vel:.4f})")
    print(f"Total Z Acceleration (used in simulation): {total_z_accel_for_print:.4f}") 
    print("-------------------------------------------------")

    # --- Physics-Based Trajectory Simulation (Euler's Method) ---
    predicted_trajectory = []
    current_pos = np.array([initial_x_pos, initial_y_pos, initial_z_pos])
    current_vel = np.array([initial_x_vel, initial_y_vel, initial_z_vel])
    
    if prediction_total_duration_s <= 0:
        print(f"DEBUG: Prediction duration is non-positive ({prediction_total_duration_s:.4f}s). Returning only initial point.")
        return [tuple(current_pos)]

    dt = prediction_total_duration_s / num_steps

    for i in range(num_steps):
        # Stop simulation if Z position is at or below 0
        if current_pos[2] <= 0:
            predicted_trajectory.append(tuple(current_pos)) # Add the point where it hits the plane
            break

        predicted_trajectory.append(tuple(current_pos))

        # Calculate forces
        F_gravity = np.array([0.0, 0.0, gravity_z * PICKLEBALL_MASS_KG])

        # Directional Drag Force Calculation
        F_drag = np.array([0.0, 0.0, 0.0])

        # Horizontal velocity components
        v_x, v_y = current_vel[0], current_vel[1]
        v_horizontal_mag = np.linalg.norm([v_x, v_y])

        if v_horizontal_mag > 1e-6:
            # Drag for horizontal movement
            F_drag_horizontal_magnitude = 0.5 * air_density * (v_horizontal_mag**2) * PICKLEBALL_CROSS_SECTIONAL_AREA_M2 * drag_coefficient_horizontal
            F_drag[0] = -F_drag_horizontal_magnitude * (v_x / v_horizontal_mag)
            F_drag[1] = -F_drag_horizontal_magnitude * (v_y / v_horizontal_mag)

        # Vertical velocity component
        v_z = current_vel[2]
        v_vertical_mag = abs(v_z) # Magnitude of vertical velocity

        if v_vertical_mag > 1e-6:
            # Drag for vertical movement
            F_drag_vertical_magnitude = 0.5 * air_density * (v_vertical_mag**2) * PICKLEBALL_CROSS_SECTIONAL_AREA_M2 * drag_coefficient_vertical
            F_drag[2] = -F_drag_vertical_magnitude * np.sign(v_z) # Apply drag opposite to vertical motion
        
        F_magnus = np.array([0.0, 0.0, 0.0]) # Placeholder for Magnus force (requires spin data)

        F_net = F_gravity + F_drag + F_magnus
        
        current_accel = F_net / PICKLEBALL_MASS_KG
        
        # No non-gravitational acceleration terms from polynomial fit are added here,
        # as they are assumed to be zero due to the linear fit.

        # Euler integration: Update velocity and position
        current_vel += current_accel * dt
        current_pos += current_vel * dt

    # If the loop finished without hitting Z=0 (e.g., prediction_total_duration_s was too short)
    # ensure the last calculated point is included. This is mostly for completeness,
    # as the break condition handles the primary stopping.
    if current_pos[2] > 0 and len(predicted_trajectory) < num_steps + 1:
        predicted_trajectory.append(tuple(current_pos))

    return predicted_trajectory
# ------------------------------------------------------------------------------------------

# --- Matplotlib Plotting Logic ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Initial view
initial_elev = 30
initial_azim = 45
ax.view_init(elev=initial_elev, azim=initial_azim)

# Plot all historical points
xs_plot = [p[0] for p in plottable_points_with_timestamps]
ys_plot = [p[1] for p in plottable_points_with_timestamps]
zs_plot = [p[2] for p in plottable_points_with_timestamps]
t_values_ms_plot = [p[3] for p in plottable_points_with_timestamps]
relative_t_s_plot = [(t - t_values_ms_plot[0]) / 1000.0 for t in t_values_ms_plot]

scatter = None
cbar = None
line = None
title_text = ax.set_title('') # Placeholder for dynamic title
endpoint_curve_line = None # Initialize for global scope
parabolic_fit_line = None # Initialize for global scope
target_zone_circle = None # Initialize for global scope

def update_plot(val=None):
    global scatter, cbar, line, title_text, endpoint_curve_line, parabolic_fit_line, target_zone_circle

    current_drag_coefficient_horizontal = INITIAL_DRAG_COEFFICIENT_HORIZONTAL
    current_drag_coefficient_vertical = INITIAL_DRAG_COEFFICIENT_VERTICAL
    current_gravity_z = INITIAL_GRAVITY_ACCELERATION_Z 

    # Clear previous plot elements
    if scatter:
        scatter.remove()
    # Clear all previous trajectory lines
    for artist in ax.lines:
        artist.remove()
    if cbar:
        cbar.remove()
    
    # Re-plot historical points
    if plottable_points_with_timestamps:
        min_timestamp = min(p[3] for p in plottable_points_with_timestamps)
        max_timestamp = max(p[3] for p in plottable_points_with_timestamps)
        
        if max_timestamp == min_timestamp:
            colors = [0.5] * len(plottable_points_with_timestamps)
        else:
            colors = [(t - min_timestamp) / (max_timestamp - min_timestamp) for t in (p[3] for p in plottable_points_with_timestamps)]
        
        scatter = ax.scatter(xs_plot, ys_plot, zs_plot, c=colors, cmap='viridis', marker='o', s=20, label='Observed Points')
        cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Timestamp (Normalized)')
        
    else:
        scatter = ax.scatter([], [], [], c='r', marker='o', s=20) # Plot empty if no points

    # --- Parabolic Fit to Observed Points ---
    if len(plottable_points_with_timestamps) >= 3: # Need at least 3 points for a quadratic fit
        # Fit X, Y, Z coordinates independently as quadratic functions of relative time
        poly_coeffs_x_obs = np.polyfit(relative_t_s_plot, xs_plot, 2)
        poly_coeffs_y_obs = np.polyfit(relative_t_s_plot, ys_plot, 2)
        poly_coeffs_z_obs = np.polyfit(relative_t_s_plot, zs_plot, 2)

        # Create polynomial functions
        poly_x_obs = np.poly1d(poly_coeffs_x_obs)
        poly_y_obs = np.poly1d(poly_coeffs_y_obs)
        poly_z_obs = np.poly1d(poly_coeffs_z_obs)

        # Generate points for the fitted curve over the observed time range
        t_fit_obs = np.linspace(min(relative_t_s_plot), max(relative_t_s_plot), 100)
        x_fit_obs = poly_x_obs(t_fit_obs)
        y_fit_obs = poly_y_obs(t_fit_obs)
        z_fit_obs = poly_z_obs(t_fit_obs)

        parabolic_fit_line, = ax.plot(x_fit_obs, y_fit_obs, z_fit_obs, color='green', linestyle='-', linewidth=2, label='Parabolic Fit (Observed)')
    else:
        print("INFO: Not enough points for parabolic fit to observed data (need at least 3).")


    all_predicted_trajectories = []
    endpoints_of_trajectories = []

    # Loop from 5 to 16 points for initial kinematics
    for num_points_for_kinematics_iter in range(5, 17): # Range is exclusive at end, so 17 for 16 points
        if len(plottable_points_with_timestamps) >= num_points_for_kinematics_iter:
            points_for_fitting = plottable_points_with_timestamps[:num_points_for_kinematics_iter]
            
            # Set a sufficiently long duration to ensure it hits Z=0 if it will.
            prediction_total_duration_s = 5.0 

            current_predicted_trajectory = predict_trajectory(
                points_for_fitting,
                current_gravity_z, 
                prediction_total_duration_s,
                TRAJECTORY_PREDICTION_STEPS,
                NUM_RECENT_POINTS_FOR_EXTRA_WEIGHT,
                EXTRA_WEIGHT_MULTIPLIER,
                current_drag_coefficient_horizontal, 
                current_drag_coefficient_vertical,   
                AIR_DENSITY_KG_M3
            )
            all_predicted_trajectories.append(current_predicted_trajectory)
            
            # Store the last point (endpoint) of each trajectory
            if current_predicted_trajectory:
                endpoints_of_trajectories.append(current_predicted_trajectory[-1])
        else:
            print(f"INFO: Not enough points for NUM_POINTS_FOR_INITIAL_KINEMATICS = {num_points_for_kinematics_iter}. Skipping.")
            break # Stop if we don't have enough data points

    # Filter endpoints: remove any endpoints where Y-coordinate is less than -6.0 (more than 6m in front)
    # This filter is now applied to the predicted endpoints as requested.
    filtered_endpoints = [ep for ep in endpoints_of_trajectories if ep[1] >= -6.0]
    endpoints_of_trajectories = filtered_endpoints # Update the list with filtered endpoints

    # Plot all generated trajectories
    for i, traj in enumerate(all_predicted_trajectories):
        if traj:
            traj_xs = [p[0] for p in traj]
            traj_ys = [p[1] for p in traj]
            traj_zs = [p[2] for p in traj]
            # Use a different color for each trajectory or a gradient if many
            ax.plot(traj_xs, traj_ys, traj_zs, color=plt.cm.jet(i / len(all_predicted_trajectories)), linestyle='--', linewidth=1, alpha=0.7)

    # Draw the curve connecting endpoints
    if endpoints_of_trajectories:
        # Add (0,0,0) as the starting point for the endpoint curve
        endpoint_coords_x = [0.0] + [p[0] for p in endpoints_of_trajectories]
        endpoint_coords_y = [0.0] + [p[1] for p in endpoints_of_trajectories]
        # Force Z-coordinates to be 0.0 for the endpoint curve
        endpoint_coords_z = [0.0] * (len(endpoints_of_trajectories) + 1) # All points on Z=0 plane
        
        endpoint_curve_line, = ax.plot(endpoint_coords_x, endpoint_coords_y, endpoint_coords_z, 
                                     color='blue', linestyle='-', linewidth=3, label='Endpoint Curve')
        ax.legend() # Update legend to include endpoint curve

        # --- Draw Variable Radius Circle around the last endpoint ---
        CIRCLE_RADIUS = 0.2 # meters, fixed radius for the target zone
        last_filtered_endpoint = endpoints_of_trajectories[-1]
        center_x, center_y = last_filtered_endpoint[0], last_filtered_endpoint[1]

        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = center_x + CIRCLE_RADIUS * np.cos(theta)
        circle_y = center_y + CIRCLE_RADIUS * np.sin(theta)
        circle_z = np.zeros_like(theta) # Always on Z=0 plane

        target_zone_circle, = ax.plot(circle_x, circle_y, circle_z, color='purple', linestyle='--', linewidth=2, label='Target Zone')
        ax.legend() # Update legend again to include the target zone
    else:
        print("INFO: No filtered endpoints to draw the endpoint curve or target zone.")


    # Set consistent axis limits for better visualization
    all_x_coords = xs_plot[:]
    all_y_coords = ys_plot[:]
    all_z_coords = zs_plot[:]

    if 'x_fit_obs' in locals(): # Check if parabolic fit was generated
        all_x_coords.extend(x_fit_obs)
        all_y_coords.extend(y_fit_obs)
        all_z_coords.extend(z_fit_obs)

    for traj in all_predicted_trajectories:
        if traj:
            all_x_coords.extend([p[0] for p in traj])
            all_y_coords.extend([p[1] for p in traj])
            all_z_coords.extend([p[2] for p in traj])

    if endpoints_of_trajectories:
        all_x_coords.extend(endpoint_coords_x)
        all_y_coords.extend(endpoint_coords_y)
        all_z_coords.extend(endpoint_coords_z) # These are all 0.0

    if 'circle_x' in locals(): # Check if circle was generated
        all_x_coords.extend(circle_x)
        all_y_coords.extend(circle_y)
        all_z_coords.extend(circle_z) # These are all 0.0

    if all_x_coords: # Only set limits if there's data
        min_x, max_x = min(all_x_coords), max(all_x_coords)
        min_y, max_y = min(all_y_coords), max(all_y_coords)
        min_z, max_z = min(all_z_coords), max(all_z_coords)

        buffer = 0.5
        ax.set_xlim([min_x - buffer, max_x + buffer])
        ax.set_ylim([min_y - buffer, max_y + buffer])
        ax.set_zlim([min_z - buffer, max_z + buffer]) 

    title_text.set_text(f'Trajectory Analysis (Drag H: {current_drag_coefficient_horizontal:.2f}, Drag V: {current_drag_coefficient_vertical:.2f}, Gravity: {current_gravity_z:.2f})')
    fig.canvas.draw_idle()

# Initial plot update
update_plot()

plt.show()
