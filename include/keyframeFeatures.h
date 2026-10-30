#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <vector>
namespace superVO {
// Immutable extraction-time data. Rows refer to the original measured pixels,
// independently of later tracking, reserve merges, or optimized landmark IDs.
struct KeyframeFeatures {
    std::vector<cv::Point2f> pixels, rightPixels;
    cv::Mat descriptors;             // Nx256 CV_32F, owned
    std::vector<cv::Point3f> points; // extraction-camera coordinates
    std::vector<unsigned char> depthValid;
};
}
