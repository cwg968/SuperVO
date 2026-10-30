#pragma once
#include <frame.h>
#include <sensorTypes.h>
#include <visualMeasurements.h>
#include <array>
#include <deque>

namespace superVO {
struct LocalMapReference {
    int sourceFrame = 0;
    cv::Matx44d worldFromReference = cv::Matx44d::eye();
    std::vector<cv::Point3f> points;
    std::vector<StereoObservation> observations;
    std::vector<int> births;
    std::vector<LandmarkId> ids;
};
struct LocalMapCandidate : LocalMapReference {
    std::array<PointTracks, 2> tracks;
};
// Immutable, bounded keyframe templates survive loss from the active LK chain.
// Candidates contain measured image matches, never just projected coordinates.
class LocalMap {
  public:
    LocalMap(size_t capacity = 5, int maxAge = 20);
    void remember(const StereoImages& images, const std::vector<cv::Point3f>& points,
                  const std::vector<StereoObservation>& observations,
                  const std::vector<int>& births, const cv::Matx44d& worldFromReference,
                  int frameIndex, const std::vector<LandmarkId>& ids);
    std::vector<int> referenceFrames() const;
    void updateReferencePose(int frameIndex, const cv::Matx44d& pose);
    void updateLandmarks(const std::vector<LandmarkSeed>& optimized);
    void protectReference(int frameIndex) {
        protectedFrame_ = frameIndex;
    }
    size_t size() const {
        return frames_.size();
    }

  private:
    friend class TemplateMatcher;
    struct Frame {
        StereoImages images;
        LocalMapReference map;
    };
    size_t capacity_;
    int maxAge_;
    int protectedFrame_ = -1;
    std::deque<Frame> frames_;
};
}
