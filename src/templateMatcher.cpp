#include <templateMatcher.h>
#include <opencv2/calib3d.hpp>
namespace superVO {
std::vector<LocalMapCandidate> TemplateMatcher::match(const LocalMap& store,
                                                      const StereoImages& current,
                                                      const cv::Matx44d& prediction,
                                                      const cv::Mat& K, const cv::Matx44d& E,
                                                      int index, size_t* trackingCalls) {
    std::vector<LocalMapCandidate> result;
    if (trackingCalls)
        *trackingCalls = 0;
    for (auto it = store.frames_.rbegin(); it != store.frames_.rend(); ++it) {
        const bool protectedReference = it->map.sourceFrame == store.protectedFrame_;
        if (index <= it->map.sourceFrame ||
            (!protectedReference && index - it->map.sourceFrame > store.maxAge_))
            continue;
        LocalMapCandidate candidate;
        static_cast<LocalMapReference&>(candidate) = it->map;
        const auto transform = prediction.inv(cv::DECOMP_SVD) * candidate.worldFromReference;
        cv::Matx33d rotation;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                rotation(r, c) = transform(r, c);
        cv::Vec3d axis;
        cv::Rodrigues(rotation, axis);
        // Translation-only LK patches are not reliable across large viewpoint changes.
        if (cv::norm(axis) > .6)
            continue;
        for (int c = 0; c < 2; ++c) {
            const auto& origin = c ? it->images.right : it->images.left;
            const auto& image = c ? current.right : current.left;
            const auto projections =
                projectLandmarks(candidate.points, c ? E * transform : transform, K, image.size());
            std::vector<cv::Point2f> pixels, predicted;
            std::vector<size_t> indices;
            for (size_t j = 0; j < candidate.points.size(); ++j) {
                if (!candidate.observations[j].valid[c] || index < candidate.births[j] ||
                    (!protectedReference && index - candidate.births[j] > store.maxAge_))
                    continue;
                pixels.push_back(candidate.observations[j].pixel[c]);
                predicted.push_back(projections[j]);
                indices.push_back(j);
            }
            if (trackingCalls)
                ++*trackingCalls;
            candidate.tracks[c] = trackGuidedPoints(origin, image, pixels, predicted);
            for (auto& j : candidate.tracks[c].sourceIndices)
                j = indices.at(j);
        }
        if (!candidate.tracks[0].pixels.empty() || !candidate.tracks[1].pixels.empty())
            result.push_back(std::move(candidate));
    }
    return result;
}
}
