#ifndef TAGDETECTOR_H
#define TAGDETECTOR_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <apriltag/apriltag.h>
#include <apriltag/apriltag_pose.h>
#include <apriltag/tag36h11.h>

class tagDetector
{
  public:
    cv::Mat mDrawnFrameImg;
    Eigen::Isometry3d mse3;

  private:
    apriltag_family_t *mtf;
    apriltag_detector_t *mtd;
    apriltag_detection_info_t minfo;

    std::string mInfoType = "[TAGDETECTOR]-->";
    
  public:
    tagDetector(const cv::Mat K, const cv::Mat D);
    bool detect(cv::Mat &frameImg);
    ~tagDetector();
};



#endif