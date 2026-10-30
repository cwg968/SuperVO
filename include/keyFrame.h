#pragma once
#include <frame.h>
#include <sensorTypes.h>
#include <vector>
#include <keyframeFeatures.h>
namespace superVO {
class SuperPointDetector;
struct KeyframeData {
    struct StereoStats {
        int left = 0, right = 0, matches = 0, geometry = 0, refined = 0, refinementRejected = 0,
            refinementFallback = 0;
    } stereoStats;
    std::vector<StereoObservation> observations;
    std::vector<cv::Point3f> points;
    std::vector<int> birthFrames;
    std::vector<LandmarkId> ids; // 0 until accepted; retained landmarks keep their identity
    int retained = 0, added = 0;
    cv::Matx44d T_world_camera = cv::Matx44d::eye();
    std::shared_ptr<const KeyframeFeatures> features;
};
bool createKeyframe(KeyframeData& frame, const StereoImages& images, SuperPointDetector& detector,
                    const cv::Mat& K, double baseline, const cv::Matx44d& T_world_camera,
                    int minPoints, int maxPoints, float threshold, int frameIndex = 0,
                    bool retainFeatures = false);
// Candidate-only merge; tracked points are expressed in the old reference camera.
void addTrackedReserve(KeyframeData& frame,
                       const std::vector<StereoObservation>& trackedObservations,
                       const std::vector<cv::Point3f>& trackedPoints,
                       const std::vector<int>& birthFrames, const std::vector<LandmarkId>& ids,
                       const cv::Matx44d& currentFromReference, const cv::Mat& K,
                       const cv::Matx44d& rightFromLeft, int frameIndex, int maxAge, int maxPoints);
void assignLandmarkIds(KeyframeData& frame, LandmarkId& nextId);
}
