#pragma once
#include <inertialEstimator.h>
#include <windowEstimator.h>
namespace superVO {
// One estimator per run. The legacy filter is a baseline, never a second
// correction stage on top of a window using the same pixels and IMU samples.
class VioEstimator {
  public:
    VioEstimator(StereoCalibration camera, ImuCalibration imu, VioOptions options);
    ~VioEstimator();
    VioEstimator(const VioEstimator&);
    VioEstimator& operator=(const VioEstimator&);
    bool initialized() const;
    bool usesPixels() const;
    void initialize(const InertialState&, TimestampNs, int frame);
    void propagate(const std::vector<ImuSample>& interval);
    bool correct(const cv::Matx44d& currentFromReference, int reference, double rmse,
                 size_t inliers, const VisualMeasurements&);
    void seed(const VisualMeasurements&);
    bool correctStationary(const cv::Vec3d& angularVelocity, const cv::Vec3d& acceleration);
    void rememberReference(int frame);
    void protectReference(int frame);
    std::optional<cv::Matx44d> referencePose(int frame) const;
    std::pair<double, double> relativeUncertainty(int frame) const;
    std::vector<LandmarkSeed> optimizedLandmarks() const;
    InertialState state() const;
    WindowDiagnostics diagnostics() const;
    EstimationFeedback feedback() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
