#pragma once
#include <sensorTypes.h>
#include <vector>

namespace superVO {
struct MapLandmark {
    LandmarkId id = 0;
    cv::Point3d world;
};
// Owned display data, expressed in the same odometry world as T_world_rect.
// Image coordinates are actual current left-image measurements, never projections.
struct VisualizationSnapshot {
    TimestampNs timestampNs = 0;
    std::vector<cv::Point2f> measuredLeft, extractedLeft;
    std::vector<MapLandmark> landmarks;
};
}
