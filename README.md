Person Detection for ROS
======================================================

Description
------------
This project is a machine learning SVM-HoG-based person detection for ROS (C++). It extends the basic person detection of PCL by a spatio-temporal filtering to increase robustness and reliability. The research was published at the IEEE MFI 2015, so I would be glad if you reference it in case of using it for academic work. It was originally developed for ROS Hydro, and requires using PCL and OpenCV, and was tested with the Microsoft Kinect camera. Having installed the required dependencies, it can be run via 'rosrun person_detection person_detection'. Make sure that the camera topics are running to subscribe to the image and point cloud data.

The flowchart below shows the algorithmic solution for the RGB-D based hybrid person detection.
<img src ="https://github.com/sebastianstarke/person_detection/blob/master/images/flowchart.png" width="100%">

The following images demonstrate the performance of the algorithm.<br/>
<img src ="https://github.com/sebastianstarke/person_detection/blob/master/images/1.png" width="45%">
<img src ="https://github.com/sebastianstarke/person_detection/blob/master/images/2.png" width="45%">
<img src ="https://github.com/sebastianstarke/person_detection/blob/master/images/3.png" width="45%">
<img src ="https://github.com/sebastianstarke/person_detection/blob/master/images/4.png" width="45%">
