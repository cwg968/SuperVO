#include <stereoMatching.h>
#include <opencv2/imgproc.hpp>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace superVO {
std::vector<cv::DMatch> matchSuperPointDescriptors(const cv::Mat& left, const cv::Mat& right,
                                                   float threshold) {
    if (!std::isfinite(threshold) || threshold <= 0 || threshold > 2)
        throw std::runtime_error("Invalid descriptor distance threshold");
    auto validate = [](const cv::Mat& matrix) {
        if (matrix.empty())
            return std::vector<bool>{};
        if (matrix.dims != 2 || matrix.type() != CV_32F || matrix.cols != 256 ||
            !cv::checkRange(matrix))
            throw std::runtime_error("Expected finite Nx256 float descriptors");
        std::vector<bool> valid(matrix.rows);
        for (int i = 0; i < matrix.rows; ++i) {
            const double n = cv::norm(matrix.row(i));
            if (n > 1e-12 && std::abs(n - 1) > 1e-3)
                throw std::runtime_error("Descriptors must be unit normalized");
            valid[i] = n > 1e-12;
        }
        return valid;
    };
    auto validLeft = validate(left), validRight = validate(right);
    if (left.empty() || right.empty())
        return {};
    cv::Mat similarity;
    cv::gemm(left, right, 1, cv::noArray(), 0, similarity, cv::GEMM_2_T);
    std::vector<int> bestLeft(left.rows, -1), bestRight(right.rows, -1);
    std::vector<float> scoreLeft(left.rows, -std::numeric_limits<float>::infinity());
    std::vector<float> scoreRight(right.rows, -std::numeric_limits<float>::infinity());
    for (int i = 0; i < left.rows; ++i)
        if (validLeft[i])
            for (int j = 0; j < right.rows; ++j)
                if (validRight[j]) {
                    const float score = similarity.at<float>(i, j);
                    if (score > scoreLeft[i]) {
                        scoreLeft[i] = score;
                        bestLeft[i] = j;
                    }
                    if (score > scoreRight[j]) {
                        scoreRight[j] = score;
                        bestRight[j] = i;
                    }
                }
    std::vector<cv::DMatch> matches;
    for (int i = 0; i < left.rows; ++i) {
        int j = bestLeft[i];
        if (j < 0 || bestRight[j] != i)
            continue;
        const float distance =
            std::sqrt(std::max(0.f, 2 - 2 * std::clamp(scoreLeft[i], -1.f, 1.f)));
        if (distance < threshold)
            matches.emplace_back(i, j, distance);
    }
    return matches;
}
namespace {
constexpr int patchRadius = 5, searchRadius = 3;
constexpr float minStdDev = 3.f, minCorrelation = .8f, uniquenessMargin = .02f;
struct SearchResult {
    bool valid = false;
    float x = 0, cost = 1;
};
bool patchInside(const cv::Mat& image, cv::Point2f p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && p.x >= patchRadius + 1 &&
           p.y >= patchRadius + 1 && p.x <= image.cols - patchRadius - 2 &&
           p.y <= image.rows - patchRadius - 2;
}
bool normalizedPatch(const cv::Mat& image, cv::Point2f p, cv::Mat& patch) {
    if (!patchInside(image, p))
        return false;
    cv::getRectSubPix(image, {2 * patchRadius + 1, 2 * patchRadius + 1}, p, patch, CV_32F);
    cv::Scalar mean, deviation;
    cv::meanStdDev(patch, mean, deviation);
    if (deviation[0] < minStdDev)
        return false;
    patch -= mean[0];
    patch /= deviation[0] * std::sqrt(double(patch.total()));
    return true;
}
SearchResult searchEpipolar(const cv::Mat& reference, const cv::Mat& current, cv::Point2f origin,
                            cv::Point2f seed) {
    SearchResult result;
    cv::Mat patch;
    if (!normalizedPatch(reference, origin, patch) ||
        !patchInside(current, {seed.x - searchRadius, origin.y}) ||
        !patchInside(current, {seed.x + searchRadius, origin.y}))
        return result;
    auto cost = [&](float x) {
        cv::Mat candidate;
        if (!normalizedPatch(current, {x, origin.y}, candidate))
            return 2.f;
        return float(1 - std::clamp(patch.dot(candidate), -1., 1.));
    };
    std::array<float, 2 * searchRadius + 1> costs{};
    int best = 0;
    for (int i = 0; i < int(costs.size()); ++i) {
        costs[i] = cost(seed.x + i - searchRadius);
        if (costs[i] < costs[best])
            best = i;
    }
    if (best == 0 || best == int(costs.size()) - 1 || costs[best] > 1 - minCorrelation)
        return result;
    float second = 2;
    for (int i = 0; i < int(costs.size()); ++i)
        if (std::abs(i - best) > 1)
            second = std::min(second, costs[i]);
    const float curvature = costs[best - 1] + costs[best + 1] - 2 * costs[best];
    if (second - costs[best] < uniquenessMargin || curvature < 1e-4f)
        return result;
    result.x = seed.x + best - searchRadius;
    result.cost = costs[best];
    const float offset = .5f * (costs[best - 1] - costs[best + 1]) / curvature;
    if (!std::isfinite(offset) || std::abs(offset) > 1)
        return result;
    auto consider = [&](float x, float value) {
        if (value < result.cost) {
            result.x = x;
            result.cost = value;
        }
    };
    const float center = result.x;
    consider(center + offset, cost(center + offset));
    // Minimize the actual interpolated ZNCC cost, rather than trusting only
    // a quadratic fitted to integer samples. The bounded interval stays local.
    constexpr float ratio = .61803398875f;
    float a = center - 1, b = center + 1, u = b - ratio * (b - a), v = a + ratio * (b - a),
          costU = cost(u), costV = cost(v);
    for (int i = 0; i < 10; ++i) {
        if (costU < costV) {
            b = v;
            v = u;
            costV = costU;
            u = b - ratio * (b - a);
            costU = cost(u);
        } else {
            a = u;
            u = v;
            costU = costV;
            v = a + ratio * (b - a);
            costV = cost(v);
        }
    }
    consider(u, costU);
    consider(v, costV);
    result.valid = result.cost <= 1 - minCorrelation;
    return result;
}
}
EpipolarRefinement refineEpipolarStereo(const cv::Mat& left, const cv::Mat& right,
                                        cv::Point2f pixel, cv::Point2f seed) {
    if (left.empty() || right.empty() || left.type() != CV_8UC1 || right.type() != CV_8UC1 ||
        left.size() != right.size())
        throw std::runtime_error(
            "Epipolar refinement expects equally sized rectified grayscale images");
    EpipolarRefinement result;
    result.right = seed;
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) || !std::isfinite(seed.x) ||
        !std::isfinite(seed.y) || pixel.x - seed.x <= .5 || std::abs(pixel.y - seed.y) > 1)
        return result;
    const auto forward = searchEpipolar(left, right, pixel, seed);
    if (!forward.valid || pixel.x - forward.x <= .5)
        return result;
    const cv::Point2f refined(forward.x, pixel.y);
    const auto backward = searchEpipolar(right, left, refined, pixel);
    if (!backward.valid)
        return result;
    result.forwardBackward = std::abs(backward.x - pixel.x);
    if (result.forwardBackward > .5f)
        return result;
    result.valid = true;
    result.right = refined;
    result.correlation = 1 - forward.cost;
    return result;
}

}
