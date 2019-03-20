# 2D SLAM with Extended Kalman Filter

The given code implements the 2D feature based EKF-SLAM on the popular Victoria Park Dataset. The odometry is obtained for dead-reckoning. Measurements are obtained from GPS and LIDARS, which provides range and bearing measurements of trees that are processed to provide point features. Main processing code is in slam.py, and the supporting code is available in slam_utils and tree_extraction.
