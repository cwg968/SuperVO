#pragma once
#include <imuTypes.h>
#include <visualMeasurements.h>
#include <vioOptions.h>
#include <windowAudit.h>
#include <memory>
#include <optional>
#include <string>

namespace superVO {
struct WindowDiagnostics {
    int states = 0, landmarks = 0, observations = 0, inliers = 0, iterations = 0, priorRank = 0;
    int marginalizedStates = 0, marginalizedLandmarks = 0;
    double initialCost = 0, finalCost = 0;
    bool accepted = false;
    int informativeLandmarks = 0;
    int posteriorOutliers = 0, removedObservations = 0;
    bool cleanupAttempted = false;
    int cleanupIterations = 0;
    double cleanupInitialCost = 0, cleanupFinalCost = 0;
    // 0 accepted; 1 no attempt; 2 insufficient history; 3 solve rejected; 4 geometry rejected;
    // 5 cleanup solve rejected; 6 cleanup geometry rejected.
    int rejection = 1;
    bool stationaryAttempted = false;
    double stationaryPosteriorSpeed = -1;
    double stationaryInnovation = -1;
    double imuGyroNoise = -1, imuAccelNoise = -1;
    std::string solverMessage;
    std::optional<WindowAudit> audit;
};
// Independent estimator: no detector, image processing, dataset, file or runtime
// dependencies. Measurement covariances are supplied by the visual frontend.
class WindowEstimator {
  public:
    WindowEstimator(StereoCalibration camera, ImuCalibration imu, WindowOptions options = {});
    ~WindowEstimator();
    WindowEstimator(const WindowEstimator&);
    WindowEstimator& operator=(const WindowEstimator&);
    bool initialized() const;
    void initialize(const InertialState& state, TimestampNs timestamp, int frame);
    void propagate(const std::vector<ImuSample>& interval);
    bool correct(const VisualMeasurements& measurements);
    void seed(const VisualMeasurements& measurements);
    bool correctStationary();
    void rememberReference(int frame);
    void protectReference(int frame);
    std::optional<cv::Matx44d> referencePose(int frame) const;
    std::optional<cv::Point3d> landmarkPosition(LandmarkId id) const;
    std::vector<LandmarkSeed> optimizedLandmarks() const;
    std::pair<double, double> relativeUncertainty(int reference) const;
    InertialState state() const;
    WindowDiagnostics diagnostics() const;
    EstimationFeedback feedback() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
