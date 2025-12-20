import cv2
import numpy as np
import math
import time
import serial
import serial.tools.list_ports
from pupil_apriltags import Detector
import pyrealsense2 as rs
import sys
import threading
import select

# --- Global state variables for communication between threads ---
running_state = False
quit_flag = False
telemetry_enabled = True
state_lock = threading.Lock()
command_buffer = ""

def is_data():
    """Checks if there's data to be read from stdin without blocking."""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def input_thread():
    """
    A separate thread to handle non-blocking terminal input.
    """
    global running_state, quit_flag, telemetry_enabled, command_buffer
    
    while not quit_flag:
        if is_data():
            char = sys.stdin.read(1)
            
            if char == '\n':
                command = command_buffer.strip().lower()
                
                with state_lock:
                    if command == 's':
                        running_state = True
                        print("\nRobot state set to RUNNING.")
                    elif command == 'p':
                        running_state = False
                        print("\nRobot state set to STOPPED.")
                    elif command == 't':
                        telemetry_enabled = not telemetry_enabled
                        print(f"\nTelemetry output is now {'ON' if telemetry_enabled else 'OFF'}.")
                    elif command == 'q':
                        print("\nQuit command received. Exiting...")
                        quit_flag = True
                    else:
                        print(f"\nUnknown command: '{command}'")
                
                command_buffer = ""
            else:
                command_buffer += char
        
        time.sleep(0.01)

# --- End of global state variables and input thread ---

# --------- Camera setup (global shutter for AprilTag detection) ------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera.")

cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 90)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
MANUAL_EXPOSURE_VALUE = 700
cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE_VALUE)

camera_matrix = np.array([
    [370.073, 0., 369.735],
    [0., 444.601, 260.813],
    [0., 0., 1.]
])
dist_coeffs = np.array([0.00572082, -0.00537963, -0.00265946, -0.00365685, -0.00619617])

fx = camera_matrix[0, 0]
fy = camera_matrix[1, 1]
cx = camera_matrix[0, 2]
cy = camera_matrix[1, 2]

tag_size = 0.146

detector = Detector(families='tag25h9', nthreads=16, quad_decimate=3.0, quad_sigma=0.0,
                    refine_edges=1, decode_sharpening=0.25, debug=0)

# ----------- RealSense T265 setup -----------------
print("Initializing RealSense T265...")
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.pose)
pipeline.start(config)
print("RealSense T265 pipeline started.")

# ----------- Serial setup to Arduino ----------------
ser = None
is_connected = False

def find_arduino(port_hint=None):
    print("Scanning serial ports for Arduino...")
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
    else:
        print("Found ports:")
        for port in ports:
            print(f" - {port.device}: {port.description}")
            arduino_hwids = [
                "1A86:7523", "2341:0043", "2341:0001", "2341:0042",
                "2341:003d", "2341:005a", "2341:0036", "2341:0055",
                "1B4F:9206", "10C4:EA60", "2A03:0043",
            ]
            if "Arduino" in port.description or any(hwid in port.hwid for hwid in arduino_hwids):
                if port_hint is None or port.device == port_hint:
                    print(f"Arduino found on {port.device}")
                    return port.device
    print("Arduino not found.")
    return None

def find_and_connect_arduino():
    global ser, is_connected
    if ser and ser.is_open:
        ser.close()
    arduino_port = find_arduino()
    if arduino_port:
        try:
            ser = serial.Serial(arduino_port, 115200, timeout=1)
            time.sleep(3)
            print("Successfully connected to Arduino.")
            is_connected = True
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to Arduino on {arduino_port}: {e}")
            is_connected = False
            return False
    else:
        is_connected = False
        return False

# ----------- Static transform from Camera to T265 (Robot Base) -------------
T_t265_cam = np.array([
    [1, 0, 0, 0.0],
    [0, 1, 0, 0.037],
    [0, 0, 1, 0.0],
    [0, 0, 0, 1]
])

def pose_to_transform(t, R):
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.reshape(3)
    return T

def transform_to_pose(T):
    t = T[0:3, 3]
    yaw = -math.atan2(T[2, 0], T[2, 2])
    return t, yaw

def wrap_angle(angle):
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    if wrapped < 0:
        wrapped += 2 * math.pi
    return wrapped

# --- FLAWED LOGIC: Applying linear deltas without accounting for rotation ---
def apply_delta_pose(x, z, yaw, dx, dz, dyaw):
    # This simplified version ignores rotation and only adds linear deltas.
    return x + dx, z + dz, yaw + dyaw
# --------------------------------------------------------------------------

def rotation_matrix_from_quaternion(qw, qx, qy, qz):
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

fused_x, fused_z, fused_yaw = 0.0, 0.0, 0.0
prev_t265_x, prev_t265_z, prev_t265_yaw = None, None, None

MAX_ROBOT_POWER = 255
POWER_SCALE_FACTOR = 200

def get_apriltag_pose(results):
    if len(results) > 1:
        with state_lock:
            if telemetry_enabled:
                print("Error, more than one AprilTag detected, cannot determine unique pose.")
        return None
    elif len(results) == 1:
        r = results[0]
        if r.pose_t is not None and r.pose_R is not None:
            t_camera_from_tag = r.pose_t.flatten()
            R_camera_from_tag = r.pose_R
            x_raw, y_raw, z_raw = t_camera_from_tag
            # The yaw from the raw pose is not used in this simplified version
            return x_raw, z_raw, 0
    return None

def send_drive_command(angle_deg, power):
    global is_connected, ser
    if not is_connected or ser is None or not ser.is_open:
        with state_lock:
            if telemetry_enabled:
                print("Serial port is not connected. Command not sent.")
        return
    power = max(0, min(MAX_ROBOT_POWER, int(power)))
    command = f"{angle_deg},{power}\n"
    try:
        ser.write(command.encode())
    except serial.SerialException as e:
        with state_lock:
            if telemetry_enabled:
                print(f"Error writing to serial port: {e}. Connection lost.")
        is_connected = False

# Start the input thread
input_thread_handle = threading.Thread(target=input_thread, daemon=True)
input_thread_handle.start()

# --- Main loop ---
try:
    if not find_and_connect_arduino():
        print("Initial connection to Arduino failed. Robot control unavailable until a connection is established.")

    print("\n--- Terminal Control Enabled ---")
    print("Commands: s (start), p (stop), t (telemetry), q (quit)")
    print("Type a command and press Enter.")
    print("--------------------------------\n")

    while True:
        with state_lock:
            if quit_flag:
                break

        if not is_connected:
            print("Attempting to reconnect to Arduino...")
            if not find_and_connect_arduino():
                time.sleep(1)
                continue

        ret, frame = cap.read()
        if not ret:
            with state_lock:
                if telemetry_enabled:
                    print("Warning: Could not read frame from camera. Continuing loop...")
            time.sleep(0.1)
            continue

        try:
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            with state_lock:
                if telemetry_enabled:
                    print(f"OpenCV processing error: {e}. Skipping frame.")
            time.sleep(0.1)
            continue

        results = detector.detect(gray, estimate_tag_pose=True,
                                  camera_params=[fx, fy, cx, cy],
                                  tag_size=tag_size)

        apriltag_pose = get_apriltag_pose(results)

        frames = pipeline.wait_for_frames()
        pose_frame = frames.get_pose_frame()

        if pose_frame:
            pose_data = pose_frame.get_pose_data()
            t265_x = pose_data.translation.x
            t265_z = pose_data.translation.z
            t265_yaw = -math.atan2(pose_data.rotation.y, pose_data.rotation.w)

            if prev_t265_x is None:
                prev_t265_x, prev_t265_z, prev_t265_yaw = t265_x, t265_z, t265_yaw
                fused_x, fused_z, fused_yaw = t265_x, t265_z, t265_yaw
                with state_lock:
                    if telemetry_enabled:
                        print(f"Initial T265 Pose: X={fused_x:.2f}m, Z={fused_z:.2f}m")

            dx = t265_x - prev_t265_x
            dz = t265_z - prev_t265_z
            prev_t265_x, prev_t265_z, prev_t265_yaw = t265_x, t265_z, t265_yaw

            if apriltag_pose is not None:
                fused_x, fused_z, fused_yaw = apriltag_pose
                with state_lock:
                    if telemetry_enabled:
                        print(f"AprilTag-based Pose: X={fused_x:.2f}m, Z={fused_z:.2f}m")
            else:
                # --- STRAFE LOGIC ---
                fused_x += dx
                fused_z += dz
                with state_lock:
                    if telemetry_enabled:
                        print(f"T265-based Pose: X={fused_x:.2f}m, Z={fused_z:.2f}m")

        else:
            with state_lock:
                if telemetry_enabled:
                    print("No T265 pose data available. Continuing loop...")

        with state_lock:
            current_running_state = running_state
        
        if current_running_state and is_connected:
            target_x, target_z = 0.0, 1.0
            error_x = target_x - fused_x
            error_z = fused_z - target_z
            target_power = int(POWER_SCALE_FACTOR * math.hypot(error_x, error_z))
            strafing_angle_rad = math.atan2(error_x, error_z)
            strafing_angle_deg = math.degrees(wrap_angle(strafing_angle_rad))
            send_drive_command(strafing_angle_deg, target_power)
        elif is_connected:
            send_drive_command(0, 0)

        with state_lock:
            state_text = "RUNNING" if running_state else "STOPPED"
        state_color = (0, 255, 0) if running_state else (0, 0, 255)
        connection_text = "CONNECTED" if is_connected else "DISCONNECTED"
        connection_color = (0, 255, 0) if is_connected else (0, 0, 255)

        try:
            cv2.putText(undistorted, f"State: {state_text}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
            cv2.putText(undistorted, f"Serial: {connection_text}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, connection_color, 2)
            for r in results:
                corners = r.corners.astype(int)
                cv2.polylines(undistorted, [corners], True, (0, 255, 0), 2)
                center = tuple(int(c) for c in r.center)
                cv2.circle(undistorted, center, 5, (0, 0, 255), -1)
                cv2.putText(undistorted, f"ID: {r.tag_id}", (center[0]-10, center[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('AprilTag Detection', undistorted)
            if cv2.waitKey(1) & 0xFF == ord('~'):
                pass
        except cv2.error as e:
            pass

        time.sleep(0.01)

finally:
    if ser and ser.is_open:
        print("Sending stop command to Arduino...")
        send_drive_command(0, 0)
        time.sleep(0.1)
    
    if ser and ser.is_open:
        ser.close()
    pipeline.stop()
    cap.release()
    cv2.destroyAllWindows()
