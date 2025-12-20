import pyzed.sl as sl

# Create a Camera object
zed = sl.Camera()

# Initialize with default parameters
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
init_params.coordinate_units = sl.UNIT.METER

status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    exit(1)

# Enable positional tracking
tracking_params = sl.PositionalTrackingParameters()
zed.enable_positional_tracking(tracking_params)

# Runtime parameters
runtime_parameters = sl.RuntimeParameters()

# Loop
while True:
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        pose = sl.Pose()
        zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        translation = pose.get_translation()
        print(f"Camera position: {translation.get()}")  # x, y, z in meters
