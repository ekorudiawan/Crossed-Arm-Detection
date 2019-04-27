# Crossed-Arm-Detection

#### Overview

There are two files and one sub folder in directory sources :
1. cross-arm-hough-transfrom.py : This file is source code of generalized hough transform implementaion for detecting cross arm.
2. cross-arm-optical-flow.py : This file is source code of optical flow implementation for detecting cross arm.
3. temp_img : This is a subfolder for storing template image of generalized hough transfrom algorithm. 

#### Dependencies

This source code implemented in Python 3.7 that required depencies :

* Numpy
* OpenCV version 3.46-dev 

Running cross-arm-hough-transform.py will caused an error, because It needs modified version of OpenCV that enable python wrapper for Generalized Hough Algorithm. Cross-arm-optical-flow.py can be run normally in same version OpenCV. 

#### Running with Docker Container

To avoid some mistake, I create a Dockerfile of my environment. First step running with docker is building docker image for this docker file. To build the image just call docker_build.bat file from Windows terminal. Running program with Docker needs to install X11 Server in your computer. In Windows, I used https://sourceforge.net/projects/vcxsrv/

```
docker_build.bat
```

Then to run the program with Docker, just call docker_run.bat from Windows terminal.

```
docker_run.bat
```

When running with docker, by default program generalized hough transform will run. To run optical flow program type this command in Windows terminal

```
docker run -it --rm -e DISPLAY=10.0.75.1:0.0 ekorudiawan/python-opencv python ./sources/cross-arm-optical-flow.py
```

