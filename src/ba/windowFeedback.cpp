#include "windowInternal.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace superVO {
EstimationFeedback WindowEstimator::Impl::classifyObservations(bool updateHistory) {
    EstimationFeedback result;
    result.frame = current().frame;
    for (auto& entry : points)
        for (auto& o : entry.second.observations) {
            const auto f = projection(o.pixel);
            const auto& x = nodes.at(o.frame).x;
            const auto p =
                (f.R_camera_imu * (ba::quaternion(x.data()).conjugate() *
                                   (Eigen::Map<const Eigen::Vector3d>(entry.second.x.data()) -
                                    ba::vector(x.data(), 4))) +
                 f.t_camera_imu)
                    .eval();
            double pixel = std::numeric_limits<double>::infinity(), white = pixel;
            if (p.allFinite() && p.z() > .02) {
                const Eigen::Vector2d r(f.fx * p.x() / p.z() + f.cx - o.pixel.pixel.x,
                                        f.fy * p.y() / p.z() + f.cy - o.pixel.pixel.y);
                pixel = r.norm();
                white = (f.W * r).squaredNorm();
            }
            // chi-square(2), 99%; the raw 3px cap agrees with the existing
            // acceptance gate. Whitened covariance is not counted a second time.
            const bool outlier = !std::isfinite(white) || white > 9.21 || pixel >= 3.;
            if (updateHistory && o.lastEvaluatedFrame != current().frame) {
                o.outlierStreak = outlier ? std::min(2u, o.outlierStreak + 1) : 0;
                o.lastEvaluatedFrame = current().frame;
            }
            result.observations.push_back(
                {entry.first, o.frame, o.pixel.camera, pixel, white,
                 outlier ? ObservationState::Outlier : ObservationState::Inlier});
        }
    return result;
}
void WindowEstimator::Impl::prunePersistentOutliers(EstimationFeedback& feedback) {
    using Key = std::tuple<LandmarkId, int, int>;
    struct Candidate {
        double error;
        Key key;
        size_t feedbackIndex;
    };
    std::vector<Candidate> candidates;
    const size_t count = feedback.observations.size();
    for (size_t k = 0; k < count; ++k) {
        const auto& f = feedback.observations[k];
        const auto& observations = points.at(f.landmark).observations;
        const auto it = std::find_if(observations.begin(), observations.end(), [&](const auto& o) {
            return o.frame == f.frame && o.pixel.camera == f.camera;
        });
        if (it != observations.end() && f.state == ObservationState::Outlier &&
            it->outlierStreak + unsigned(it->lastEvaluatedFrame != current().frame) >= 2)
            candidates.push_back({f.whitenedSquaredError, {f.landmark, f.frame, f.camera}, k});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.error != b.error ? a.error > b.error : a.key < b.key;
    });
    // Bounded maintenance. Preserve at least a stereo pair / two constraints
    // per landmark; never marginalize discarded outliers into the prior.
    const size_t budget = std::min(size_t(32), count / 10);
    size_t removed = 0;
    for (const auto& c : candidates) {
        if (removed >= budget)
            break;
        auto& observations = points.at(std::get<0>(c.key)).observations;
        if (observations.size() <= 2)
            continue;
        const auto it = std::find_if(observations.begin(), observations.end(), [&](const auto& o) {
            return o.frame == std::get<1>(c.key) && o.pixel.camera == std::get<2>(c.key);
        });
        if (it == observations.end())
            continue;
        observations.erase(it);
        ++removed;
        feedback.observations[c.feedbackIndex].state = ObservationState::Removed;
    }
    diagnostic.removedObservations = removed;
    diagnostic.posteriorOutliers =
        std::count_if(feedback.observations.begin(), feedback.observations.end(),
                      [](const auto& f) { return f.state != ObservationState::Inlier; });
    if (removed)
        covariance.reset();
}
}
