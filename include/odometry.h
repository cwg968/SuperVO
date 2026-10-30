#pragma once
#include <sensorTypes.h>
#include <imuTypes.h>
#include <trackingTypes.h>
#include <recoveryPolicy.h>
#include <stationaryDetector.h>
#include <windowEstimator.h>
#include <vioOptions.h>
#include <stereoBaEstimator.h>
#include <visualizationSnapshot.h>
#include <memory>
#include <optional>
namespace superVO {
struct TrackingResult {
    TimestampNs timestampNs = 0;
    int frameIndex = 0;
    TrackingStatus status = TrackingStatus::InitializationFailed;
    std::optional<cv::Matx44d> T_world_rect;
    FrameDiagnostics diagnostics;
    std::optional<InertialState> inertial;
    bool inertialUpdate = false, inertialReseed = false;
    double inertialMs = 0;
    StationaryEvidence stationary;
    bool zeroVelocityUpdate = false;
    RecoveryPhase recoveryPhase = RecoveryPhase::Tracking;
    double sinceConstraintSeconds = 0;
    std::optional<VisualPoseCandidate> visual;
    WindowDiagnostics backend;
    StereoBaDiagnostics stereoBa;
    double stereoBaMs = 0;
    int baReferenceFrame = 0;
    std::vector<BaPoseUpdate> baPoses;
    std::shared_ptr<const KeyframeFeatures> keyframeFeatures;
};
struct OdometrySnapshot {
    int acceptedFrames = 0;
    TimestampNs timestampNs = 0;
    cv::Matx44d T_world_rect = cv::Matx44d::eye();
    std::vector<LandmarkId> activeIds;
};
// Detector must outlive this estimator. No dataset, filesystem or playback dependency.
class Odometry {
  public:
    Odometry(StereoCalibration calibration, TrackingOptions options, SuperPointDetector& detector,
             std::optional<ImuCalibration> imu = std::nullopt, VioOptions vio = {},
             StereoBaOptions ba = {});
    ~Odometry();
    Odometry(const Odometry&) = delete;
    Odometry& operator=(const Odometry&) = delete;
    TrackingResult process(const StereoFrame& frame, const std::vector<ImuSample>& imu = {});
    OdometrySnapshot snapshot() const;
    VisualizationSnapshot visualization() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
