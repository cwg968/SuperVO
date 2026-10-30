#include <motionPrediction.h>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <stdexcept>
namespace superVO {
cv::Matx44d predictCameraPose(const cv::Matx44d& before, const cv::Matx44d& last,
                              double previousSeconds, double nextSeconds) {
    if (!std::isfinite(previousSeconds) || !std::isfinite(nextSeconds) || previousSeconds <= 0 ||
        nextSeconds <= 0 || nextSeconds > .25 || nextSeconds / previousSeconds > 4 ||
        nextSeconds / previousSeconds < .25)
        return last;
    const auto delta = before.inv(cv::DECOMP_SVD) * last;
    cv::Matx33d rotation;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            rotation(r, c) = delta(r, c);
    cv::Vec3d axis;
    cv::Rodrigues(rotation, axis);
    cv::Rodrigues(axis * (nextSeconds / previousSeconds), rotation);
    auto increment = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            increment(r, c) = rotation(r, c);
        increment(r, 3) = delta(r, 3) * nextSeconds / previousSeconds;
    }
    return last * increment;
}
cv::Matx44d ConstantVelocityPredictor::predict(TimestampNs timestampNs) const {
    return predictCameraPose(before_, last_, previousSeconds_, (timestampNs - timestampNs_) * 1e-9);
}
void ConstantVelocityPredictor::accept(const cv::Matx44d& pose, TimestampNs timestampNs) {
    if (timestampNs < 0 || (initialized_ && timestampNs <= timestampNs_) ||
        !cv::checkRange(cv::Mat(pose)))
        throw std::invalid_argument("Invalid accepted motion state");
    if (initialized_)
        previousSeconds_ = (timestampNs - timestampNs_) * 1e-9;
    before_ = last_;
    last_ = pose;
    timestampNs_ = timestampNs;
    initialized_ = true;
}
void ConstantVelocityPredictor::rebase(const cv::Matx44d& correction) {
    if (!cv::checkRange(cv::Mat(correction)))
        throw std::invalid_argument("Invalid motion correction");
    before_ = correction * before_;
    last_ = correction * last_;
}
}
