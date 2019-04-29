# Install the base image - ubuntu 18.04, with CUDA 10.0 and CUDNN 7
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

# Install some dependencies
RUN apt update -y
RUN apt install python3-dev python3-pip -y

# Install tensorflow GPU, version 1.13.1
RUN pip3 install tensorflow-gpu==1.13.1

# Install Opencv version 4.1.0
RUN apt-get update -y
RUN apt-get upgrade -y
RUN apt-get install build-essential cmake unzip pkg-config -y
RUN apt-get install libjpeg-dev libpng-dev libtiff-dev -y
RUN apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y
RUN apt-get install libxvidcore-dev libx264-dev -y
RUN apt-get install libgtk-3-dev -y
RUN apt-get install libatlas-base-dev gfortran -y
RUN apt-get install python3-dev -y && apt-get install wget

WORKDIR /
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip
RUN unzip opencv.zip
RUN unzip opencv_contrib.zip
RUN mv opencv-4.1.0 opencv
RUN mv opencv_contrib-4.1.0 opencv_contrib

RUN cd /opencv  && mkdir build
RUN cd /opencv/build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D INSTALL_C_EXAMPLES=ON \
	-D OPENCV_ENABLE_NONFREE=ON \
	-D OPENCV_EXTRA_MODULES_PATH=/opencv_contrib/modules \
	-D PYTHON_EXECUTABLE=/usr/bin/python3 \
	-D BUILD_EXAMPLES=ON ..
RUN cd /opencv/build && make -j4
RUN cd /opencv/build && make install
RUN cd /opencv/build && ldconfig
