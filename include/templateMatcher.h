#pragma once
#include <localMap.h>
namespace superVO {
// Measures image correspondences against immutable templates; never changes storage.
class TemplateMatcher {
  public:
    static std::vector<LocalMapCandidate> match(const LocalMap& store, const StereoImages& current,
                                                const cv::Matx44d& predictedWorld, const cv::Mat& K,
                                                const cv::Matx44d& rightFromLeft, int frameIndex,
                                                size_t* trackingCalls = nullptr);
};
}
