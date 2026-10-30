#pragma once
#include <frame.h>
#include <vector>
namespace superVO {
// Bit mask recorded in frames.csv: interval=1, relative loss=2, reserve=4,
// rapid loss=8, poor image coverage=16, lost camera support=32.
int replenishmentReason(int framesSinceKeyframe, size_t live, size_t anchor, size_t previous,
                        int occupiedCells, int minInliers, int keyframeInterval);
int landmarkCoverage(const std::vector<cv::Point2f>& pixels, cv::Size imageSize);
std::vector<cv::Point2f> cameraPixels(const std::vector<StereoObservation>& observations,
                                      int camera);
int stereoCoverage(const std::vector<StereoObservation>& observations, cv::Size size);
}
