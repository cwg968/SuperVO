#pragma once
#include <visualMeasurements.h>
#include <memory>
#include <optional>

namespace superVO {
struct StereoBaOptions {
    bool enabled = false;
    int keyframes = 6, landmarks = 300, iterations = 6, minInliers = 20;
};
struct StereoBaDiagnostics {
    int states = 0, landmarks = 0, observations = 0, inliers = 0, iterations = 0;
    // 0 accepted, 1 not triggered, 2 insufficient connected history, 3 solver, 4 geometry.
    int rejection = 1;
    bool attempted = false, accepted = false;
    double initialCost = 0, finalCost = 0;
};
struct BaPoseUpdate {
    int frame = 0;
    cv::Matx44d worldFromCamera = cv::Matx44d::eye();
};
// Pure stereo geometry only: no detector, image storage, IMU or runtime dependency.
// Keeps keyframes and one transient current frame; only keyframes trigger a solve.
class StereoBaEstimator {
  public:
    StereoBaEstimator(StereoCalibration camera, StereoBaOptions options = {});
    ~StereoBaEstimator();
    StereoBaEstimator(const StereoBaEstimator&);
    StereoBaEstimator& operator=(const StereoBaEstimator&);
    bool update(const VisualMeasurements&, const cv::Matx44d& initialPose, bool keyframe);
    std::optional<cv::Matx44d> pose(int frame) const;
    std::vector<BaPoseUpdate> poses() const;
    std::vector<LandmarkSeed> landmarks() const;
    StereoBaDiagnostics diagnostics() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
