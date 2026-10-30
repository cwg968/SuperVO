#include <imuTypes.h>
#include <cmath>
#include <stdexcept>
namespace superVO {
void ImuCalibration::validate() const {
    for (double x : {gyroNoise, accelNoise, gyroWalk, accelWalk, rateHz})
        if (!std::isfinite(x) || x <= 0)
            throw std::invalid_argument("Invalid IMU noise/rate calibration");
    StereoCalibration check;
    check.intrinsics = cv::Mat::eye(3, 3, CV_64F);
    check.imageSize = {1, 1};
    check.baseline = 1;
    check.T_body_raw = T_imu_rect;
    check.validate();
}
void validateImuInterval(const std::vector<ImuSample>& samples, TimestampNs begin, TimestampNs end,
                         double rateHz) {
    if (begin < 0 || begin > end || samples.empty() || samples.front().timestampNs != begin ||
        samples.back().timestampNs != end || !std::isfinite(rateHz) || rateHz <= 0 ||
        (begin == end && samples.size() != 1))
        throw std::invalid_argument("Invalid IMU interval boundaries");
    for (size_t i = 0; i < samples.size(); ++i) {
        if (!cv::checkRange(cv::Mat(samples[i].acceleration)) ||
            !cv::checkRange(cv::Mat(samples[i].angularVelocity)))
            throw std::invalid_argument("Nonfinite IMU sample");
        if (i && (samples[i].timestampNs <= samples[i - 1].timestampNs ||
                  (samples[i].timestampNs - samples[i - 1].timestampNs) * 1e-9 > 4 / rateHz))
            throw std::invalid_argument("Invalid IMU interval spacing");
    }
}
}
