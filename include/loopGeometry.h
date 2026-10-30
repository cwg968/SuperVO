#pragma once
#include <keyframeFeatures.h>
#include <sensorTypes.h>
#include <optional>
namespace superVO {
struct LoopKeyframe {
    int frame = 0;
    TimestampNs timestamp = 0;
    cv::Matx44d odomPose = cv::Matx44d::eye();
    cv::Vec3d gravity{0, 0, -9.81};
    std::shared_ptr<const KeyframeFeatures> features;
};
struct LoopOptions {
    bool enabled = false, synchronous = false;
    double keyframeSeconds = 1, exclusionSeconds = 10;
    int candidates = 8, minInliers = 30;
    double pixelThreshold = 2.5;
};
struct LoopVerification {
    bool valid = false;
    int matches = 0, inliers = 0, rejection = 1;
    double rmse = 0, coverage = 0, tilt = 0;
    // Maps historical/reference camera coordinates into current camera.
    cv::Matx44d currentFromReference = cv::Matx44d::eye();
};
cv::Mat pooledSuperPoint(const cv::Mat& descriptors);
LoopVerification verifyLoop(const LoopKeyframe& reference, const LoopKeyframe& current,
                            const StereoCalibration&, const LoopOptions&);
}
