import pyrealsense2 as rs
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Configure pose pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.pose)
pipeline.start(config)

# Initialize 3D plotting
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Store trajectory
x_vals, y_vals, z_vals = [], [], []
window_size = 10  # Try 10 or higher
last_x, last_y, last_z = None, None, None
max_jump = 0.2  # meters, try lowering this

def moving_average(data, window):
    if len(data) < window:
        return data[-1]
    return np.mean(data[-window:])

try:
    while True:
        frames = pipeline.wait_for_frames()
        pose = frames.get_pose_frame()

        if pose:
            data = pose.get_pose_data()
            x = data.translation.x
            y = data.translation.y
            z = data.translation.z

            # Outlier rejection: skip if jump is too large
            if last_x is not None:
                if abs(x - last_x) > max_jump or abs(y - last_y) > max_jump or abs(z - last_z) > max_jump:
                    print("Jump detected, skipping point")
                    continue

            # Optionally, warn on high velocity
            v = np.linalg.norm([data.velocity.x, data.velocity.y, data.velocity.z])
            if v > 1.0:
                print("High velocity detected, drift likely!")

            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            last_x, last_y, last_z = x, y, z

            # Clear and update plot
            ax.clear()
            ax.plot3D(x_vals, y_vals, z_vals, 'b', label='Raw Trajectory')
            ax.scatter(x, y, z, c='r', label='Current Pos')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('T265 3D Position')
            ax.legend()
            # Set fixed axis limits: always show 1m range centered at origin
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim([-0.5, 0.5])
            ax.set_zlim([-0.5, 0.5])
            plt.draw()
            plt.pause(0.01)

except KeyboardInterrupt:
    pass

finally:
    pipeline.stop()
