#pragma once
#include <imuTypes.h>
#include <memory>
#include <optional>
namespace superVO {
class InertialInitializer {
  public:
    explicit InertialInitializer(ImuCalibration calibration);
    ~InertialInitializer();
    InertialInitializer(const InertialInitializer&);
    InertialInitializer& operator=(const InertialInitializer&);
    std::optional<InertialState> observe(const cv::Matx44d& cameraPose, TimestampNs timestamp,
                                         const std::vector<ImuSample>& interval);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Error-state filter: p,v,R,bg,ba,g and correlated camera-pose clones.
// Visual updates use relative stereo poses, never independent absolute VO poses.
class InertialEstimator {
  public:
    explicit InertialEstimator(ImuCalibration calibration);
    ~InertialEstimator();
    InertialEstimator(const InertialEstimator&);
    InertialEstimator& operator=(const InertialEstimator&);
    bool initialized() const;
    void initialize(const InertialState& initial, TimestampNs timestamp);
    void propagate(const std::vector<ImuSample>& interval);
    bool correct(const cv::Matx44d& currentFromReference, int referenceFrame, double pixelRmse,
                 size_t inliers);
    bool correctStationary(const cv::Vec3d& meanAngularVelocity, const cv::Vec3d& meanAcceleration);
    void rememberReference(int frameIndex);
    void protectReference(int frameIndex);
    // Relative camera position/rotation standard deviations, including clone cross covariance.
    std::pair<double, double> relativeUncertainty(int frameIndex) const;
    std::optional<cv::Matx44d> referencePose(int frameIndex) const;
    InertialState state() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
