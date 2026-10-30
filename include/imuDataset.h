#pragma once
#include <imuTypes.h>
#include <filesystem>
namespace superVO {
// Dataset adapter only; it never reads ground truth.
class ImuDataset {
  public:
    ImuDataset(const std::filesystem::path& mavRoot, const StereoCalibration& cameras);
    const ImuCalibration& calibration() const {
        return calibration_;
    }
    TimestampNs firstTimestamp() const {
        return samples_.front().timestampNs;
    }
    TimestampNs lastTimestamp() const {
        return samples_.back().timestampNs;
    }
    // Includes interpolated endpoints, without double-integrating shared boundaries.
    std::vector<ImuSample> interval(TimestampNs begin, TimestampNs end) const;

  private:
    ImuCalibration calibration_;
    std::vector<ImuSample> samples_;
};
}
