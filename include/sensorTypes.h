#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include <optional>

namespace superVO {
using TimestampNs = int64_t;
using LandmarkId = uint64_t;
struct StereoImages {
    cv::Mat left, right;
};
struct StereoFrame {
    TimestampNs timestampNs;
    StereoImages images;
};

// T_A_B maps coordinates in B into A. Tracking consumes rectified images.
// Body calibration is optional: KITTI does not provide an IMU/body transform.
struct StereoCalibration {
    cv::Mat intrinsics;
    cv::Size imageSize;
    double baseline = 0;
    cv::Matx44d T_rect_raw = cv::Matx44d::eye();
    std::optional<cv::Matx44d> T_body_raw;
    void validate() const;
    cv::Matx44d rightFromLeft() const {
        auto result = cv::Matx44d::eye();
        result(0, 3) = -baseline;
        return result;
    }
    cv::Matx44d rawPose(const cv::Matx44d& T_world_rect) const {
        return T_world_rect * T_rect_raw;
    }
    std::optional<cv::Matx44d> bodyFromRectified() const {
        if (!T_body_raw)
            return std::nullopt;
        return *T_body_raw * T_rect_raw.inv(cv::DECOMP_SVD);
    }
};
}
