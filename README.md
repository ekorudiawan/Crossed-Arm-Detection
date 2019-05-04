# Crossed-Arm-Detection

#### Overview

There are three files and one sub folder in directory sources :

1. cross-arm-hough-transfrom.py : This is source code of generalized hough transform implementation.
2. cross-arm-optical-flow-shi.py : This is source code of optical flow with Shi-Thomasi feature detection.
3. cross-arm-optical-flow-orb.py : This is source code of optical flow with ORB feature detection.
4. temp_img : This is a subfolder for storing template image of generalized hough transfrom algorithm.

#### Dependencies

This source code implemented in Python 3.7 that required dependencies :

* Numpy
* OpenCV version 3.46-dev

Running cross-arm-hough-transform.py will caused an error, because It needs modified version of OpenCV that enable python wrapper for Generalized Hough Algorithm. Cross-arm-optical-flow-shi.py and Cross-arm-optical-flow-orb.py can be run normally in same version of OpenCV.

#### Running with Docker Container

To avoid some mistake, I create a Dockerfile of my environment. First step running with docker is building docker image from this dockerfile. To build the image just call docker_build.bat file from Windows terminal.

```cmd
docker_build.bat
```

Normally building docker image need around 15 minutes to complete. Running program with Docker needs to install X-Server in your computer. In Windows OS can be used [VcXsrv](https://sourceforge.net/projects/vcxsrv/).

Then to run the program with Docker, just call docker_run.bat from Windows terminal.

```cmd
docker_run.bat
```

When running with docker, the default program will run generalized hough transform source code. To run optical flow program type this command in Windows terminal.

```cmd
docker run -it --rm -e DISPLAY=10.0.75.1:0.0 ekorudiawan/python-opencv python ./sources/cross-arm-optical-flow-shi.py
```

or

```cmd
docker run -it --rm -e DISPLAY=10.0.75.1:0.0 ekorudiawan/python-opencv python ./sources/cross-arm-optical-flow-orb.py
```

#### Demo Videos

Demo of this program can be seen in this link :

1. [https://www.youtube.com/watch?v=Jiz0DU1QtuM](https://www.youtube.com/watch?v=Jiz0DU1QtuM) : Demo of Generalized Hough Transform
2. [https://www.youtube.com/watch?v=NNTj_hrmefI](https://www.youtube.com/watch?v=NNTj_hrmefI) : Demo of Optical Flow with Shi-Thomasi Feature Detector
3. [https://www.youtube.com/watch?v=LEHlKfXk1ds](https://www.youtube.com/watch?v=LEHlKfXk1ds) : Demo of Optical Flow with Shi-Thomasi Feature Detector
