REM docker run -it --rm -e DISPLAY=10.0.75.1:0.0 -v D:/Source-Code/Computer-Vision/Tugas-Computer-Vision/Crossed-Arm-Detection:/home/ ekorudiawan/python-opencv bash
docker run -it --rm -e DISPLAY=10.0.75.1:0.0 ekorudiawan/python-opencv 
REM docker run -it --rm -e DISPLAY=10.0.75.1:0.0 ekorudiawan/python-opencv python ./sources/cross-arm-optical-flow-shi.py
REM docker run -it --rm -e DISPLAY=10.0.75.1:0.0 ekorudiawan/python-opencv python ./sources/cross-arm-optical-flow-orb.py