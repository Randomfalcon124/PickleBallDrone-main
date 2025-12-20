Hello!

This project is the first step in an attempt to build a drone that plays pickleball. The idea is to build a drone that will take off from a pickleball court and catch balls from practice serves, so a UAV ball boy.
Currently, I have already tested 2 RGB-D cameras and various algorithms for detecting the ball including HSV filtering, RGB contour detection, circularity filtering, depth discontinuity detection, and YOLO object detection.
My next step is to record playing data from a pickleball court and then run each algorithm on the data to determine which one or which combination is best.
The step after that is to code the predictive algorithm for the ball's movement and test it on the data.
The next step is to code april tag detection for localization.
Finally, the last step is to mount the camera(s) on the drone and create a script using MAVSDK that continually feeds the Pixhawk controller on the drone coordinates to travel to based on the algorithm performed from the camera(s) input and an onboard computer.

I am making this project for two reasons: I am bad at pickleball, and I want to get better. To get better, I need to practice, but there isn't always someone to practice with. This drone is meant to solve this problem.
