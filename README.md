Hello!

This project is the first step in an attempt to build a drone that plays pickleball. The current idea is to build a drone that will take off from a pickleball court and catch balls from practice serves, so a UAV ball boy.

Currently, I have already tested multiple depth and tracking cameras and various algorithms for detecting the ball, including HSV filtering, RGB contour detection, circularity filtering, depth discontinuity detection, and YOLO object detection. For localization, I have tested VSLAM using a global shutter camera, VSLAM on the T265 tracking camera, optical flow, and April Tag localization using a global shutter camera, with the last method being the most accurate by far.

In order to test the whole workflow, I have built a small holonomic robotic drivebase that uses an Nvidia Jetson Orin Nano as a brain connected to an Arduino Nano which provides instructions to the motor controllers. It serves as a cheap teststand for all of the localization and tracking systems before I have to buy an expensive UAV.

I am making this project because I am bad at pickleball, and I want to get better. However, to get better, I need to practice, but there isn't always someone to practice with. This drone (UAV) is meant to solve this problem.

