#ifndef SUPERVO_POSE_ESTIMATION_H
#define SUPERVO_POSE_ESTIMATION_H
#include <opencv2/core.hpp>
#include <vector>
#include <array>
#include <frame.h>

namespace superVO {
struct PoseEstimate {
    bool valid = false;
    cv::Matx44d T_camera_reference = cv::Matx44d::eye();
    std::vector<int> inliers; // indices into the supplied point/pixel correspondence arrays
    double rmse = 0;
};
PoseEstimate estimatePose(const std::vector<cv::Point3f>& points,
                          const std::vector<cv::Point2f>& pixels, const cv::Mat& K,
                          int minInliers = 12, double threshold = 3.0);
// All landmarks remain in reference-left coordinates. A right-camera PnP
// returns T_right_current_reference_left, converted by inverse(T_right_left).
cv::Matx44d leftPoseFromRight(const cv::Matx44d& rightPose, const cv::Matx44d& rightFromLeft);
struct StereoPoseQuality {
    std::array<std::vector<int>, 2> inlierObservations;
    size_t uniqueInliers = 0;
    double coverage = 0, rmse = 0, score = 0;
};
struct LandmarkPoseEstimate {
    bool valid = false;
    cv::Matx44d T_camera_reference = cv::Matx44d::eye();
    std::vector<int> inliers; // unique indices into the reference landmark array
    double rmse = 0;
};
struct StereoPoseEstimate {
    LandmarkPoseEstimate pose;
    StereoPoseQuality quality;
    size_t uniqueObservations = 0;
    double conditionRatio = 0;
    // 0 accepted, 1 insufficient landmarks, 2 no solution, 3 support ratio,
    // 4 coverage, 5 ill-conditioned geometry, 6 invalid input.
    int rejection = 1;
};
StereoPoseEstimate estimateStereoPose(const std::vector<cv::Point3f>& points,
                                      const std::array<PointTracks, 2>& tracks, const cv::Mat& K,
                                      const cv::Matx44d& rightFromLeft, cv::Size imageSize,
                                      const cv::Matx44d& initialLeftPose, int minInliers = 20,
                                      double threshold = 3);
StereoPoseQuality assessStereoPose(const cv::Matx44d& leftPose,
                                   const std::vector<cv::Point3f>& points,
                                   const std::array<PointTracks, 2>& tracks, const cv::Mat& K,
                                   const cv::Matx44d& rightFromLeft, cv::Size imageSize,
                                   double threshold = 3);
// Fixed engineering margin; not a calibrated statistical confidence interval.
inline constexpr double stereoPoseScoreMargin = .02;
// Candidates must already pass their own >=minimum, >=40% PnP checks.
// Scores use exactly the same observations; each landmark contributes at most 1.
int selectStereoPose(const std::array<PoseEstimate, 2>& candidates,
                     const std::array<StereoPoseQuality, 2>& quality, int minimum);
cv::Matx44d cameraPose(const cv::Matx44d& T_world_reference, const cv::Matx44d& T_camera_reference);

}
// Compatibility entry points for the original frontend.
void poseEstimation2d2d(std::vector<cv::Point2f>, std::vector<cv::Point2f>, cv::Mat&, cv::Mat&,
                        cv::Mat&);
void poseEstimation3d2d(std::vector<cv::Point3f>, std::vector<cv::Point2f>, cv::Mat&, cv::Mat&,
                        cv::Mat&);
#endif
