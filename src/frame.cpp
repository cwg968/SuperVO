#include <frame.h>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <array>
namespace superVO {
namespace {
PointTracks trackRaw(const cv::Mat& reference, const cv::Mat& current,
                     const std::vector<cv::Point2f>& points,
                     const std::vector<cv::Point2f>& predictions = {}) {
    PointTracks result;
    if (reference.empty() || current.empty() || reference.type() != CV_8UC1 ||
        current.type() != CV_8UC1 || reference.size() != current.size())
        throw std::runtime_error("LK expects equally sized uint8 grayscale images");
    if (points.empty())
        return result;
    if (!predictions.empty() && predictions.size() != points.size())
        throw std::invalid_argument("LK prediction count mismatch");
    std::vector<cv::Point2f> forward, backward;
    std::vector<uchar> validForward, validBackward;
    std::vector<float> errorForward, errorBackward;
    forward = predictions;
    cv::calcOpticalFlowPyrLK(reference, current, points, forward, validForward, errorForward,
                             {21, 21}, 3,
                             {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, .01},
                             predictions.empty() ? 0 : cv::OPTFLOW_USE_INITIAL_FLOW);
    std::vector<cv::Point2f> candidates, origins;
    std::vector<size_t> indices;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& p = forward[i];
        if (validForward[i] && errorForward[i] < 30 && std::isfinite(p.x) && std::isfinite(p.y) &&
            p.x >= 4 && p.y >= 4 && p.x < current.cols - 4 && p.y < current.rows - 4) {
            candidates.push_back(p);
            origins.push_back(points[i]);
            indices.push_back(i);
        }
    }
    if (candidates.empty())
        return result;
    backward = origins;
    cv::calcOpticalFlowPyrLK(
        current, reference, candidates, backward, validBackward, errorBackward, {21, 21}, 3,
        {cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, .01}, cv::OPTFLOW_USE_INITIAL_FLOW);
    for (size_t i = 0; i < candidates.size(); ++i)
        if (validBackward[i] && cv::norm(backward[i] - origins[i]) <= 1.0) {
            result.pixels.push_back(candidates[i]);
            result.sourceIndices.push_back(indices[i]);
        }
    return result;
}

cv::Mat locallyNormalized(const cv::Mat& image) {
    cv::Mat values, mean, variance, deviation, result;
    image.convertTo(values, CV_32F);
    cv::GaussianBlur(values, mean, {}, 8);
    cv::GaussianBlur(values.mul(values), variance, {}, 8);
    variance -= mean.mul(mean);
    cv::max(variance, 25, variance);
    cv::sqrt(variance, deviation);
    values = (values - mean) / deviation;
    values.convertTo(result, CV_8U, 32, 128);
    return result;
}

std::array<double, 9> intensityQuantiles(const cv::Mat& image) {
    std::array<size_t, 256> histogram{};
    size_t count = 0;
    for (int y = 0; y < image.rows; y += 4)
        for (int x = 0; x < image.cols; x += 4) {
            ++histogram[image.at<uchar>(y, x)];
            ++count;
        }
    std::array<double, 9> result{};
    size_t cumulative = 0, index = 0;
    for (size_t value = 0; value < histogram.size() && index < result.size(); ++value) {
        cumulative += histogram[value];
        while (index < result.size() && cumulative * 10 >= (index + 1) * count)
            result[index++] = value;
    }
    return result;
}
}

PointTracks trackPoints(const cv::Mat& reference, const cv::Mat& current,
                        const std::vector<cv::Point2f>& points, bool compensateExposure) {
    auto original = trackRaw(reference, current, points);
    if (!compensateExposure || points.empty())
        return original;
    const auto a = intensityQuantiles(reference), b = intensityQuantiles(current);
    double sx = 0, sy = 0, sxx = 0, sxy = 0, change = 0;
    int count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        // Saturated pixels cannot identify an exposure gain reliably.
        if (a[i] <= 5 || a[i] >= 250 || b[i] <= 5 || b[i] >= 250)
            continue;
        sx += b[i];
        sy += a[i];
        sxx += b[i] * b[i];
        sxy += a[i] * b[i];
        change += std::abs(a[i] - b[i]);
        ++count;
    }
    if (count < 3 || change / count < 8 || sxx - sx * sx / count < 100)
        return original;
    const double gain = (sxy - sx * sy / count) / (sxx - sx * sx / count),
                 bias = (sy - gain * sx) / count;
    if (!std::isfinite(gain) || !std::isfinite(bias) || gain < .25 || gain > 4 ||
        std::abs(bias) > 100)
        return original;
    cv::Mat corrected;
    current.convertTo(corrected, CV_8U, gain, bias);
    auto candidate = trackRaw(reference, corrected, points);
    if (candidate.pixels.size() <= original.pixels.size())
        return original;
    candidate.exposureCompensated = true;
    candidate.exposureGain = gain;
    candidate.exposureBias = bias;
    return candidate;
}

std::vector<cv::Point2f> projectLandmarks(const std::vector<cv::Point3f>& points,
                                          const cv::Matx44d& transform, const cv::Mat& K,
                                          cv::Size size) {
    if (K.type() != CV_64F || K.rows != 3 || K.cols != 3 || size.width <= 0 || size.height <= 0)
        throw std::invalid_argument("Invalid projection calibration");
    std::vector<cv::Point2f> result(points.size(), {NAN, NAN});
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& p = points[i];
        const auto q = transform * cv::Vec4d(p.x, p.y, p.z, 1);
        if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2]) || q[2] <= 0)
            continue;
        const cv::Point2f pixel(K.at<double>(0, 0) * q[0] / q[2] + K.at<double>(0, 2),
                                K.at<double>(1, 1) * q[1] / q[2] + K.at<double>(1, 2));
        if (pixel.x >= 4 && pixel.y >= 4 && pixel.x < size.width - 4 && pixel.y < size.height - 4)
            result[i] = pixel;
    }
    return result;
}
PointTracks trackGuidedPoints(const cv::Mat& reference, const cv::Mat& current,
                              const std::vector<cv::Point2f>& points,
                              const std::vector<cv::Point2f>& predictions, bool normalizeLocally) {
    if (points.size() != predictions.size())
        throw std::invalid_argument("LK prediction count mismatch");
    if (reference.type() != CV_8UC1 || current.type() != CV_8UC1 || reference.empty() ||
        current.empty() || reference.size() != current.size())
        throw std::invalid_argument("Guided LK expects equally sized grayscale images");
    std::vector<cv::Point2f> origins, seeds;
    std::vector<size_t> indices;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto p = points[i], q = predictions[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(q.x) ||
            !std::isfinite(q.y) || p.x < 4 || p.y < 4 || p.x >= reference.cols - 4 ||
            p.y >= reference.rows - 4 || q.x < 4 || q.y < 4 || q.x >= current.cols - 4 ||
            q.y >= current.rows - 4)
            continue;
        origins.push_back(p);
        seeds.push_back(q);
        indices.push_back(i);
    }
    if (origins.empty())
        return {};
    auto tracks = trackRaw(normalizeLocally ? locallyNormalized(reference) : reference,
                           normalizeLocally ? locallyNormalized(current) : current, origins, seeds);
    for (auto& i : tracks.sourceIndices)
        i = indices.at(i);
    return tracks;
}

}
