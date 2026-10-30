#pragma once
#include <viewer.h>
#include <cmath>
#include <deque>
#include <map>

namespace superVO {
// Display history only: retired points are last-seen estimates, not a global BA map.
struct ViewerScene {
    static constexpr size_t maxLandmarks = 50000, maxPoses = 50000, maxKeyframes = 2000;
    std::map<LandmarkId, cv::Point3d> landmarks;
    std::deque<cv::Vec3d> trajectory;
    std::deque<cv::Matx44d> keyframes;

    void update(const ViewerFrame& frame) {
        if (!frame.T_world_rect)
            return;
        const auto& pose = *frame.T_world_rect;
        trajectory.push_back({pose(0, 3), pose(1, 3), pose(2, 3)});
        if (trajectory.size() > maxPoses)
            trajectory.pop_front();
        if (frame.keyframe) {
            keyframes.push_back(pose);
            if (keyframes.size() > maxKeyframes)
                keyframes.pop_front();
        }
        if (frame.geometry.timestampNs != frame.timestampNs)
            return;
        for (const auto& point : frame.geometry.landmarks)
            if (point.id && std::isfinite(point.world.x) && std::isfinite(point.world.y) &&
                std::isfinite(point.world.z))
                landmarks[point.id] = point.world;
        while (landmarks.size() > maxLandmarks)
            landmarks.erase(landmarks.begin());
    }
};
}
