# SuperVO
## Introduction
SuperVO is a stereo visual odometry based on optical flow using the SuperPoint feature. I used the official pre-trained model and convert it to `.wts` file that can be used by TensorRT.(https://github.com/magicleap/SuperPointPretrainedNetwork.git) It's architecture and strategy are similar to traditional slam, even simpler.

I refactor the backbone of SuperPoint using TensorRT， and use `C++` to rewrite the forward and post-processing part of the algorithm in the original code. In the `descriptor` part, `LibTorch` is used to rewrite the post-processing code of the descriptor, because some algorithms in torch are too complicated and inefficient to use C++ to rewrite.

Tested on a custom dataset `848 * 480`. The inference time on `NIVDIA 1050ti` is 31ms per frame. So it can be used in real scenes to run in real time at `30fps`.

<image src="assets/hjl1.gif">
<image src="assets/hjl2.gif">
<image src="assets/hjl3.gif">

## Dependencies
### C++11 Compiler
I use the new thread and chrono functionalities of C++11.

### OpenCV4
I use `OpenCV-4.5.4` to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org.
### CUDA
I use `CUDA-11.6`. Dowload and install instructions can be found at: https://developer.nvidia.com/cuda-toolkit-archive
### TensorRT
I use `TensorRT-8.4` that adapts to `CUDA-11.6`. Dowload and install instructions can be found at: https://developer.nvidia.cn/nvidia-tensorrt-download
### LibTorch
It should be installed with suitable `CUDA` version. Dowload and install instructions can be found at: https://pytorch.org/
### Eigen3
```sh
sudo apt install libeigen3-dev
```
### Sophus
Dowload and install instructions can be found at: 
https://github.com/strasdat/Sophus.git
### Pangolin
Dowload and install instructions can be found at: 
https://github.com/stevenlovegrove/Pangolin.git
### g2o
Dowload and install instructions can be found at: 
https://github.com/RainerKuemmerle/g2o.git

## Building SuperVO library and example
Clone the repository:
```sh

```
Please make sure you have installed all required dependencies. Execute:
```sh
mkdir build
cd build
make -j
```
## Example
### KITTI Dataset
Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php
1. First serialize model to plan file, it will generate a `.engine` file by TensorRT.
    ```sh
    ./bin superVO -s settings/kitti.yaml
    ```
2. Second deserialize plan file and run inference.
    ```sh
    ./bin superVO -d settings/kitti.yaml PATH_TO_SEQUENCE_FOLDER
    ```

## 
## TODO
1. The next step is fixing optimize problems, I use threads to make a optimizer, but it doesn't seem to works well. I will attempt to use deeplearning-based backend optimization methods.
2. Add a IMU to the system.
3. Solve the problem of map point fusion.

This repo is a personal project that helps me better understand SLAM. I borrowed the name of "SuperPoint", although its performance and accuracy did not reach the level of "super". Maybe one day I will make a real "SuperVO". 