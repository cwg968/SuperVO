#pragma once
#include <keyFrame.h>
#include <poseEstimation.h>
namespace superVO {
struct TrackingOptions {
    int maxPoints = 1000, minInliers = 20, keyframeInterval = 10;
    float threshold = .015f;
    bool retainFeatures = false;
};
enum class TrackingStatus {
    Tracking,
    InitializingInertial,
    PropagatingInertial,
    StationaryInertial,
    InitializationFailed,
    Lost
};
enum class PoseSource { None = -1, Left = 0, Right = 1, Stereo = 2 };
struct FrameDiagnostics {
    bool key = false, valid = true;
    size_t count = 0, inliers = 0, rawCount = 0, landmarksBefore = 0, lkAttempts = 0,
           jointInliers = 0;
    bool exposureRetry = false, exposureRecovered = false;
    double exposureGain = 1, exposureBias = 0;
    int reason = 0, stereoLandmarks = 0, retained = 0, added = 0;
    PoseSource selected = PoseSource::None;
    bool attempted = false;
    std::array<PointTracks, 2> tracks, guidedTracks;
    std::array<PoseEstimate, 2> candidates;
    StereoPoseEstimate jointEstimate;
    int stereoAttempts = 0, stereoObservationSet = -1;
    std::array<StereoPoseQuality, 2> quality;
    std::array<size_t, 2> rawCounts{}, guidedCounts{};
    std::array<bool, 2> retries{}, recoveries{}, guidedAttempted{}, guidedRecovered{};
    std::array<double, 2> gains{1, 1}, biases{};
    double disagreementM = 0, disagreementDeg = 0;
    int mapCandidates = 0, mapSource = -1;
    bool mapRecovered = false;
    double trackTime = 0, poseTime = 0, keyTime = 0;
    size_t landmarksAfter = 0, mapSize = 0;
    int cells = 0;
    KeyframeData::StereoStats stereoStats;
};
// Geometry is in reference-left coordinates; observations are in the last image.
struct TrackingReference {
    KeyframeData geometry;
    int frameIndex = 0;
    size_t anchorCount = 0;
};
struct VisualPoseCandidate {
    PoseSource source = PoseSource::None;
    cv::Matx44d T_current_reference = cv::Matx44d::eye();
    cv::Matx44d T_world_reference = cv::Matx44d::eye();
    int referenceFrame = 0;
    StereoPoseQuality quality;
    std::vector<size_t> landmarkIndices;
    std::vector<LandmarkId> landmarkIds;
};
}
