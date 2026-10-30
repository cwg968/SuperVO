#pragma once
#include <opencv2/core.hpp>
#include <vector>
namespace superVO {
// Mutual nearest neighbors over ALL descriptors, followed by an L2 distance gate.
// Rows are unit-normalized Nx256 float descriptors; zero rows are ignored.
std::vector<cv::DMatch> matchSuperPointDescriptors(const cv::Mat& left, const cv::Mat& right,
                                                   float maxDistance = .7f);
struct EpipolarRefinement {
    bool valid = false;
    cv::Point2f right;
    float correlation = 0, forwardBackward = 0;
};
// Rectified grayscale images. Refines only x around the descriptor seed;
// fixes right y to left y and rejects ambiguous/flat/occluded local matches.
EpipolarRefinement refineEpipolarStereo(const cv::Mat& left, const cv::Mat& right,
                                        cv::Point2f leftPixel, cv::Point2f rightSeed);
}
