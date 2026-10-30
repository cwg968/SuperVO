#pragma once
#include <imuTypes.h>
#include <deque>
namespace superVO {
struct StationaryEvidence {
    bool stationary = false;
    int imageTracks = 0;
    double flowMedian = 0, gyroRms = 0, accelerationStd = 0;
    cv::Vec3d meanAngularVelocity{}, meanAcceleration{};
};
// Independent 2D image motion + IMU test. No depth, dataset or ground truth dependency.
class StationaryDetector {
  public:
    StationaryEvidence observe(const StereoFrame& frame, const std::vector<ImuSample>& imu,
                               const InertialState& state, const ImuCalibration& calibration);

  private:
    StereoImages previous_;
    TimestampNs timestamp_ = -1;
    std::deque<ImuSample> samples_;
    std::deque<std::pair<TimestampNs, bool>> motion_;
};
}
