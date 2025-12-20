#!/usr/bin/env python3
print("SCRIPT STARTING...")
import pyrealsense2 as rs
print(f"RealSense library version: {rs.__version__}")
import time
import sys

def test_realsense_installation():
    print("--- RealSense Python Installation Test ---")
    print(f"Python interpreter: {sys.executable}")

    try:
        # --- Basic Module Check ---
        print("\n--- Module Information ---")
        # 1. Print pyrealsense2 version
        print(f"Pyrealsense2 version detected: {rs.__version__}")
        expected_version = "2.53.1"
        if rs.__version__ != expected_version:
            print(f"WARNING: Version mismatch! Expected {expected_version}, but found {rs.__version__}.")
            print("This could indicate an old `pyrealsense2` module is being loaded.")

        # 2. Check the path where pyrealsense2 is loaded from
        print(f"Pyrealsense2 loaded from: {rs.__file__}")

        # --- Device Detection Test ---
        print("\n--- Device Detection ---")
        ctx = rs.context()
        devices = ctx.query_devices()
        num_devices = len(devices)
        print(f"Detected {num_devices} RealSense device(s).")

        if num_devices == 0:
            print("WARNING: No RealSense devices found. Please ensure your camera is connected and powered.")
            print("         You can also try running `realsense-viewer` in a new terminal to check.")
            # We can still proceed to test pipeline creation, but it won't stream
        else:
            for i, dev in enumerate(devices):
                print(f"  Device {i+1}: Name: {dev.get_info(rs.camera_info.name)}, Serial: {dev.get_info(rs.camera_info.serial)}")


        # --- Pipeline Creation and Streaming Test ---
        print("\n--- Pipeline Test ---")
        print("Attempting to create rs.pipeline()...")
        pipeline = rs.pipeline()
        print("rs.pipeline() created successfully.")

        print("Configuring and starting stream...")
        config = rs.config()

        # You can add specific stream configurations here if needed, e.g.:
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming (this will try to auto-configure if no streams are enabled)
        pipeline.start(config)
        print("Streaming started successfully.")

        # Wait for a few frames to ensure data flow
        print("Waiting for 10 frames to confirm data...")
        for i in range(10):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if depth_frame or color_frame:
                print(f"  Frame {i+1}: Received {'Depth' if depth_frame else ''}{' & ' if depth_frame and color_frame else ''}{'Color' if color_frame else ''} frame(s).")
            else:
                print(f"  Frame {i+1}: No frames received (check camera connection/firmware).")
            time.sleep(0.05) # Small delay

        # Stop streaming
        print("Stopping stream...")
        pipeline.stop()
        print("Stream stopped.")

        print("\n--- RealSense Python test completed successfully! ---")
        print("Your `pyrealsense2` module appears to be correctly installed and functional.")

    except AttributeError as e:
        print(f"\nERROR: AttributeError: {e}")
        print("This specific error (`no attribute 'pipeline'`) strongly indicates that:")
        print("  1. The wrong `pyrealsense2` module is being loaded by Python.")
