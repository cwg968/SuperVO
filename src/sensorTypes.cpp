#include <sensorTypes.h>
#include <cmath>
#include <stdexcept>
namespace superVO {
namespace {
bool rigid(const cv::Matx44d& T) {
    cv::Matx33d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = T(r, c);
    return cv::checkRange(cv::Mat(T)) && cv::norm(cv::Mat(R.t() * R - cv::Matx33d::eye())) < 1e-4 &&
           std::abs(cv::determinant(R) - 1) < 1e-4 &&
           cv::norm(cv::Vec4d(T(3, 0), T(3, 1), T(3, 2), T(3, 3)) - cv::Vec4d(0, 0, 0, 1)) < 1e-9;
}
}
void StereoCalibration::validate() const {
    if (intrinsics.rows != 3 || intrinsics.cols != 3 || intrinsics.type() != CV_64FC1 ||
        !cv::checkRange(intrinsics) || intrinsics.at<double>(0, 0) <= 0 ||
        intrinsics.at<double>(1, 1) <= 0 || imageSize.width <= 0 || imageSize.height <= 0 ||
        !std::isfinite(baseline) || baseline <= 0 || !rigid(T_rect_raw) ||
        (T_body_raw && !rigid(*T_body_raw)))
        throw std::invalid_argument("Invalid rectified stereo calibration");
}
}
