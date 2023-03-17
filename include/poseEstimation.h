#ifndef POSEESTIMATION_H
#define POSEESITMATION_H

#include <iostream>
#include <opencv2/opencv.hpp>

void poseEstimation2d2d(std::vector<cv::Point2f> kpts1, std::vector<cv::Point2f> pts2, cv::Mat &R, cv::Mat &t, cv::Mat &cameraMatrix);

void poseEstimation3d2d(std::vector<cv::Point3f> p3d, std::vector<cv::Point2f> p2d, cv::Mat &R, cv::Mat &t, cv::Mat &cameraMatrix);
#endif