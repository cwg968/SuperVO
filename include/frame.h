#pragma once
#include <opencv2/core.hpp>
#include <vector>
namespace superVO {
// Missing observations must never be scored as measured image coordinates.
struct StereoObservation {
    cv::Point2f pixel[2];
    bool valid[2] = {false, false};
    bool operator==(const StereoObservation& other) const {
        return pixel[0] == other.pixel[0] && pixel[1] == other.pixel[1] &&
               valid[0] == other.valid[0] && valid[1] == other.valid[1];
    }
};
struct PointTracks {
    std::vector<cv::Point2f> pixels;
    std::vector<size_t> sourceIndices;
    bool exposureCompensated = false;
    double exposureGain = 1.0, exposureBias = 0.0;
};
// All returned arrays share one correspondence order. Failed LK points never reach PnP.
// Exposure compensation only proposes additional tracks; the caller must still validate geometry.
PointTracks trackPoints(const cv::Mat& reference, const cv::Mat& current,
                        const std::vector<cv::Point2f>& points, bool compensateExposure = false);
// Predictions initialize a real image match; invalid projections are skipped.
PointTracks trackGuidedPoints(const cv::Mat& reference, const cv::Mat& current,
                              const std::vector<cv::Point2f>& points,
                              const std::vector<cv::Point2f>& predictions,
                              bool normalizeLocally = true);
std::vector<cv::Point2f> projectLandmarks(const std::vector<cv::Point3f>& points,
                                          const cv::Matx44d& cameraFromReference, const cv::Mat& K,
                                          cv::Size size);
}
