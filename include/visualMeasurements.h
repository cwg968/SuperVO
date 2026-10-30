#pragma once
#include <frame.h>
#include <estimationFeedback.h>
#include <sensorTypes.h>
#include <vector>

namespace superVO {
enum class ObservationNoise { Uniform, Patch };
struct PixelMeasurement {
    LandmarkId landmark = 0;
    int camera = 0;
    cv::Point2d pixel;
    cv::Matx22d covariance = cv::Matx22d::eye();
};
struct LandmarkSeed {
    LandmarkId landmark = 0;
    cv::Point3d world;
};
// Seeds are numerical initial guesses, not additional depth measurements.
// A (frame, camera, landmark) tuple identifies one measured image observation.
struct VisualMeasurements {
    int frame = 0;
    TimestampNs timestamp = 0;
    std::vector<PixelMeasurement> pixels;
    std::vector<LandmarkSeed> seeds;
};
// A bounded anisotropic engineering noise model. This is not learned uncertainty
// and requires calibration against measured localization errors on held-out data.
cv::Matx22d pixelCovariance(const cv::Mat& image, cv::Point2f pixel, ObservationNoise model);
VisualMeasurements measuredPixels(const StereoImages& images, int frame, TimestampNs timestamp,
                                  const std::vector<LandmarkId>& ids,
                                  const std::vector<cv::Point3f>& points,
                                  const std::vector<StereoObservation>& observations,
                                  const cv::Matx44d& worldFromReference, ObservationNoise model);
void updateLandmarkGeometry(std::vector<cv::Point3f>& points, const std::vector<LandmarkId>& ids,
                            const cv::Matx44d& worldFromReference,
                            const std::vector<LandmarkSeed>& optimized);
}
