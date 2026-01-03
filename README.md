Hello!

This project is the first step in an attempt to build a drone that plays pickleball. The current idea is to build a drone that will take off from a pickleball court and catch balls from practice serves, so a UAV ball boy.

Currently, I have already tested multiple depth and tracking cameras and various algorithms for detecting the ball, including HSV filtering, RGB contour detection, circularity filtering, depth discontinuity detection, and YOLO object detection. For localization, I have tested VSLAM using a global shutter camera, VSLAM on the T265 tracking camera, optical flow, and April Tag localization using a global shutter camera, with the last method being the most accurate by far.

In order to test the whole workflow, I have built a small holonomic robotic drivebase that uses an Nvidia Jetson Orin Nano as a brain connected to an Arduino Nano which provides instructions to the motor controllers. It serves as a cheap teststand for all of the localization and tracking systems before I have to buy an expensive UAV.

I am making this project because I am bad at pickleball, and I want to get better. However, to get better, I need to practice, but there isn't always someone to practice with. This drone (UAV) is meant to solve this problem.

Key Files:

Arduino Sketch (arduinoDrivebaseControlSketch.ino)
Handles low-level motor control. It receives motion commands over serial and directly drives the robot hardware. Currently assumes that the robot has a fixed heading and takes desired movement direction and power as input.

Depth Pipeline (currentDepthDetectionPipeline.py)
Processes RGB-D camera data to detect objects and estimate their 3D position. This is the source of all ball position information used by the robot. Currently uses a recorded video to demonstrate functionality.

Trajectory (trajectory.py)
Takes the detected 3D positions and converts them into motion targets for the robot (where to go and how to move). Currently uses points received from real-world testing to demonstrate functionality.

InstaRun (instaRun.py)
Currently demonstrates the localization and movement of the drivebase as well as the remote control using a terminal on a laptop connecting via SSH. Will eventually orchestrate the system by running the depth pipeline, passing results to the trajectory module, and sending the final drive commands to the Arduino based on its localization. 

How They Will Work Together

The depth pipeline provides object positions → the trajectory module decides the robot’s motion → InstaRun sends commands → the Arduino executes them on the hardware.
