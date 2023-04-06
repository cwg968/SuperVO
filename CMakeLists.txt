cmake_minimum_required(VERSION 3.16)
project(demo_superVO)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)

set(Torch_DIR "/home/cwg/libtorch/share/cmake/Torch")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.6/bin/nvcc")
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV 4 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(apriltag REQUIRED)
find_package(Sophus REQUIRED)
find_package(Torch REQUIRED)
find_package(realsense2 REQUIRED)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)
find_package(Glog REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME}
            src/trtEngine.cpp
            src/poseEstimation.cpp
            src/frame.cpp
            src/keyFrame.cpp
            src/mapDrawer.cpp
            src/tagDetector.cpp
            src/mapPoint.cpp
            src/map.cpp
            src/optimizer.cpp
            src/dataLoader.cpp
            src/tracker.cpp)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${OpenCV_INCLUDE_DIRS}
                    ${Pangolin_INCLUDE_DIRS}
                    ${apriltag_INCLUDE_DIRS}
                    ${Sophus_INCLUDE_DIRS}
                    ${realsense2_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIRS}
                    ${CSPARSE_INCLUDE_DIR}
                    ${GLOG_INCLUDE_DIRS}
                    "/usr/include/eigen3"
                    "/usr/local/cuda/include"
                    "/usr/include/x86_64-linux-gnu/"
                    "/home/cwg/libtorch/include/torch/csrc/api/include/"
                    "/home/cwg/libtorch/include/torch/csrc/autograd/"
                    "/home/cwg/libtorch/include")

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/lib/x86_64-linux-gnu/)

target_link_libraries(${PROJECT_NAME}
                        ${OpenCV_LIBS}
                        ${Pangolin_LIBRARIES}
                        ${apriltag_LIBS}
                        ${TORCH_LIBRARIES}
                        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
                        Sophus::Sophus
                        realsense2
                        nvinfer
                        cudart
                        pthread
                        fmt::fmt
                        ${CSPARSE_LIBRARY}
                        ${GLOG_LIBRARIES}
                        )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(superVO superVO.cpp)
target_link_libraries(superVO ${PROJECT_NAME})





