#include <localMap.h>
#include <opencv2/calib3d.hpp>
#include <stdexcept>
#include <algorithm>

namespace superVO {
void LocalMap::updateLandmarks(const std::vector<LandmarkSeed>& optimized) {
    for (auto& frame : frames_)
        updateLandmarkGeometry(frame.map.points, frame.map.ids, frame.map.worldFromReference,
                               optimized);
}
LocalMap::LocalMap(size_t capacity, int maxAge) : capacity_(capacity), maxAge_(maxAge) {
    if (capacity < 1 || maxAge < 1)
        throw std::invalid_argument("Invalid local map bounds");
}
void LocalMap::remember(const StereoImages& images, const std::vector<cv::Point3f>& points,
                        const std::vector<StereoObservation>& observations,
                        const std::vector<int>& births, const cv::Matx44d& worldFromReference,
                        int index, const std::vector<LandmarkId>& ids) {
    if (points.size() != ids.size() || points.size() != observations.size() ||
        points.size() != births.size() || images.left.empty() || images.right.empty() ||
        images.left.type() != CV_8UC1 || images.right.type() != CV_8UC1 ||
        images.left.size() != images.right.size() || !cv::checkRange(cv::Mat(worldFromReference)) ||
        (!frames_.empty() && index <= frames_.back().map.sourceFrame))
        throw std::invalid_argument("Invalid local map keyframe");
    Frame frame;
    frame.images = {images.left.clone(), images.right.clone()};
    frame.map.sourceFrame = index;
    frame.map.worldFromReference = worldFromReference;
    frame.map.ids = ids;
    frame.map.points = points;
    frame.map.observations = observations;
    frame.map.births = births;
    frames_.push_back(std::move(frame));
    for (auto it = frames_.begin(); it != frames_.end();) {
        if (it->map.sourceFrame != protectedFrame_ && index - it->map.sourceFrame > maxAge_)
            it = frames_.erase(it);
        else
            ++it;
    }
    while (frames_.size() > capacity_) {
        auto oldest = std::find_if(frames_.begin(), frames_.end(), [&](const Frame& f) {
            return f.map.sourceFrame != protectedFrame_;
        });
        frames_.erase(oldest);
    }
}
std::vector<int> LocalMap::referenceFrames() const {
    std::vector<int> result;
    for (const auto& frame : frames_)
        result.push_back(frame.map.sourceFrame);
    return result;
}
void LocalMap::updateReferencePose(int index, const cv::Matx44d& pose) {
    for (auto& frame : frames_)
        if (frame.map.sourceFrame == index) {
            frame.map.worldFromReference = pose;
            return;
        }
}
}
