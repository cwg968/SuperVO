#include <visualMeasurements.h>
#include <map>
#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

namespace superVO {
cv::Matx22d pixelCovariance(const cv::Mat& image, cv::Point2f p, ObservationNoise model) {
    if (image.type() != CV_8UC1 || image.empty() || !std::isfinite(p.x) || !std::isfinite(p.y))
        throw std::invalid_argument("Pixel noise requires a finite pixel and grayscale image");
    if (model == ObservationNoise::Uniform)
        return cv::Matx22d::eye();
    const int x = cvRound(p.x), y = cvRound(p.y), radius = 4;
    if (x < radius + 1 || y < radius + 1 || x >= image.cols - radius - 1 ||
        y >= image.rows - radius - 1)
        return cv::Matx22d::eye() * 9.;
    cv::Matx22d H = cv::Matx22d::zeros();
    for (int v = y - radius; v <= y + radius; ++v)
        for (int u = x - radius; u <= x + radius; ++u) {
            const double gx = .5 * (double(image.at<uchar>(v, u + 1)) - image.at<uchar>(v, u - 1));
            const double gy = .5 * (double(image.at<uchar>(v + 1, u)) - image.at<uchar>(v - 1, u));
            H += cv::Matx22d(gx * gx, gx * gy, gx * gy, gy * gy);
        }
    cv::Mat values, vectors;
    cv::eigen(cv::Mat(H), values, vectors);
    cv::Matx22d C = cv::Matx22d::zeros();
    for (int k = 0; k < 2; ++k) {
        // 0.5 px floor covers interpolation/model mismatch. Weak directions
        // receive up to 3 px std; detector scores are deliberately unused.
        const double variance = std::min(9., .25 + 400. / std::max(1., values.at<double>(k)));
        const double a = vectors.at<double>(k, 0), b = vectors.at<double>(k, 1);
        C += variance * cv::Matx22d(a * a, a * b, a * b, b * b);
    }
    return C;
}
VisualMeasurements measuredPixels(const StereoImages& images, int frame, TimestampNs timestamp,
                                  const std::vector<LandmarkId>& ids,
                                  const std::vector<cv::Point3f>& points,
                                  const std::vector<StereoObservation>& observations,
                                  const cv::Matx44d& T, ObservationNoise model) {
    if (ids.size() != points.size() || ids.size() != observations.size() || timestamp < 0)
        throw std::invalid_argument("Inconsistent visual measurement arrays");
    VisualMeasurements batch;
    batch.frame = frame;
    batch.timestamp = timestamp;
    std::unordered_set<LandmarkId> unique;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!ids[i] || !unique.insert(ids[i]).second)
            throw std::invalid_argument("Visual measurements require unique accepted landmark IDs");
        const auto p = T * cv::Vec4d(points[i].x, points[i].y, points[i].z, 1);
        if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
            continue;
        bool measured = false;
        for (int c = 0; c < 2; ++c) {
            if (!observations[i].valid[c])
                continue;
            const auto uv = observations[i].pixel[c];
            const auto& image = c ? images.right : images.left;
            if (!std::isfinite(uv.x) || !std::isfinite(uv.y) || uv.x < 0 || uv.y < 0 ||
                uv.x >= image.cols || uv.y >= image.rows)
                throw std::invalid_argument("Visual measurement is outside its image");
            batch.pixels.push_back({ids[i], c, uv, pixelCovariance(image, uv, model)});
            measured = true;
        }
        if (measured)
            batch.seeds.push_back({ids[i], {p[0], p[1], p[2]}});
    }
    return batch;
}
void updateLandmarkGeometry(std::vector<cv::Point3f>& points, const std::vector<LandmarkId>& ids,
                            const cv::Matx44d& worldFromReference,
                            const std::vector<LandmarkSeed>& optimized) {
    if (points.size() != ids.size())
        throw std::invalid_argument("Geometry/identity mismatch");
    std::unordered_map<LandmarkId, cv::Point3d> lookup;
    for (const auto& p : optimized)
        lookup.emplace(p.landmark, p.world);
    const auto T = worldFromReference.inv(cv::DECOMP_SVD);
    for (size_t k = 0; k < ids.size(); ++k)
        if (auto it = lookup.find(ids[k]); it != lookup.end()) {
            const auto& p = it->second;
            const auto q = T * cv::Vec4d(p.x, p.y, p.z, 1);
            points[k] = {float(q[0]), float(q[1]), float(q[2])};
        }
}
void discardPosteriorOutliers(int frame, const std::vector<LandmarkId>& ids,
                              std::vector<StereoObservation>& pixels,
                              const EstimationFeedback& feedback) {
    if (ids.size() != pixels.size())
        throw std::invalid_argument("Feedback IDs and observations must align");
    if (feedback.frame != frame)
        return; // Never apply stale history to the current image.
    std::map<LandmarkId, size_t> index;
    for (size_t k = 0; k < ids.size(); ++k)
        index.emplace(ids[k], k);
    for (const auto& o : feedback.observations)
        if (o.frame == frame && o.camera >= 0 && o.camera < 2 &&
            o.state != ObservationState::Inlier) {
            const auto it = index.find(o.landmark);
            if (it != index.end())
                pixels[it->second].valid[o.camera] = false;
        }
}
}
