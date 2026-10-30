#pragma once
#include <trackingTypes.h>
#include <localMap.h>
#include <optional>
namespace superVO {
struct VisualTrackingResult {
    TrackingReference reference;
    std::optional<VisualPoseCandidate> candidate;
    FrameDiagnostics diagnostics;
};
// Prediction initializes measured matching only. The caller owns state acceptance.
VisualTrackingResult trackVisualFrame(const StereoImages& previous, const StereoImages& current,
                                      const TrackingReference& reference, const LocalMap& templates,
                                      const cv::Matx44d& predictedWorld,
                                      const StereoCalibration& calibration,
                                      const TrackingOptions& options, int frameIndex);
}
