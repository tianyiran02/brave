# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Assignment Comments

### FP.5
#### Way-off cases

First thing is removal of outlier points. Because the shape of objects might vary, its really hard to distinguish real object reflection and outlier. Imagine a traillor hock. It may create "outlier" pattern points, but it's real signal. 

Besides, considering the slope road. Lidar may fail to detect object down in the hill, or make false-positive detecion on road surface. For curved one, special algorithms need to take road curvature into account. Otherwise it cannot detect objects after corner.

Last thing maybe, still about outlier points. Even though the point cloud can be easily clustered together, still, use which distance to do the calculation might be pretty critical. It has directly impacts on the system accuracy. The closest points might be noisy, the further points is more accurate but the TTC calcuation will give a smaller result, which might leads to disaster. Car engineering is about life. Life matters.

### FP.6
#### Combination Test
According to mid-term project result, to make best use of time, I'll only test following combination:

- Detector: SIFT, BRISK, FAST
- Descriptor: SIFT, BRIEF, FREAK
- Searching: BF
- Selector: KNN

Rational: base on the previous test, SIFT, BRISK and FAST are most unique algorithms. Other algorithms more or less similar to those 3. Besides, SIFT basically represent the most tradition method, where BRISK are more advance, speed-up whole process without gradient calculation. And FAST is just something jump out of box... As for Descriptor, the dicision are similar. The tradition BF searching method was decided and KNN selector was used. Other approach is not tested because I assume the result is predictable, therefore skip those algorithm such as FLEANN and NN selector to save time.

Besides, the performance data, such as points matched and execution time, is not log and analyzed here as previous mid project has already complete this task. Current project doesn't have any optimal regards of the algorithms, repeat this analyze is not worthy of time. However, the TTC compute with each algorihtm combination for each frame is recorded for analyze.

Result shown in the excel sheet within the workspace.

#### Combination Test Result Comments
According to the result, all different camera algorihtms provide similar TTC calculations. BRIEF extractor has highest standard deviation compare with FREAK and SIFT, where FREAK and SIFT are similar. 

Because the lidar measurement is not ground true, I didn't compare the camera TTC accuracy.

#### Way-off cases

Fisrt about object detection. If model failed to detect an object, let's say a padestrain on wheel chair, then the whole system will fail on camera based TTC calcuation. Or the camera failed to crop the whole object from the image, then objects might be incomplete and TTC may not really help.

Then, as for key point detection. Shadow might also causes problem. Shadow is not properties of the object, but the environment. The key point detection for sure be influenced by shadow. The algorihtm need to remove those interference.

Besides, if the object doesn't have much edges/keypoints, for example, a grey object that almost merged with the road color, the system may fail to detect it, or calculate the distance.