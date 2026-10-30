#pragma once
#include <sensorTypes.h>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace superVO {
struct StereoSample {
    int64_t timestampNs;
    std::filesystem::path left, right;
};
class StereoDataset {
  public:
    explicit StereoDataset(const std::filesystem::path& path, const std::string& format = "auto",
                           double imageScale = 1., const std::string& kittiCameras = "auto");
    StereoImages read(size_t index) const;
    const std::vector<StereoSample>& samples() const {
        return samples_;
    }
    const cv::Mat& K() const {
        return K_;
    }
    cv::Size size() const {
        return size_;
    }
    cv::Size sourceSize() const {
        return sourceSize_;
    }
    int leftCamera() const {
        return leftCamera_;
    }
    const cv::Matx44d& cam0FromLeft() const {
        return T_cam0_left_;
    }
    double baseline() const {
        return baseline_;
    }
    // Transform between the rectified images consumed by tracking, not raw cam0/cam1.
    cv::Matx44d rightFromLeft() const {
        auto T = cv::Matx44d::eye();
        T(0, 3) = -baseline_;
        return T;
    }
    StereoCalibration calibration() const;
    const cv::Matx33d& rectification() const {
        return R_rect_;
    }
    const std::filesystem::path& root() const {
        return root_;
    }
    const std::string& format() const {
        return format_;
    }
    size_t unpairedLeft() const {
        return unpairedLeft_;
    }
    size_t unpairedRight() const {
        return unpairedRight_;
    }

  private:
    void loadEuroc();
    void loadKitti();
    std::filesystem::path root_;
    std::string format_;
    std::vector<StereoSample> samples_;
    cv::Mat K_, maps_[4];
    cv::Size size_, sourceSize_;
    double imageScale_ = 1.;
    std::string kittiCameras_;
    int leftCamera_ = 0;
    double baseline_ = 0;
    size_t unpairedLeft_ = 0, unpairedRight_ = 0;
    cv::Matx44d T_body_camera_ = cv::Matx44d::eye();
    cv::Matx33d R_rect_ = cv::Matx33d::eye();
    cv::Matx44d T_cam0_left_ = cv::Matx44d::eye();
};
}
