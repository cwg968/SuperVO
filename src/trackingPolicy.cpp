#include <trackingPolicy.h>
#include <array>
#include <algorithm>
#include <cmath>
namespace superVO {
int landmarkCoverage(const std::vector<cv::Point2f>& pixels, cv::Size size) {
    if (size.width <= 0 || size.height <= 0)
        return 0;
    std::array<bool, 24> occupied{};
    for (const auto& p : pixels) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 || p.x >= size.width ||
            p.y >= size.height)
            continue;
        occupied[int(p.y * 4 / size.height) * 6 + int(p.x * 6 / size.width)] = true;
    }
    return int(std::count(occupied.begin(), occupied.end(), true));
}
int replenishmentReason(int elapsed, size_t live, size_t anchor, size_t previous, int cells,
                        int minimum, int interval) {
    int reason = 0;
    if (elapsed >= interval)
        reason |= 1;
    if (live < .65 * anchor)
        reason |= 2;
    if (live < static_cast<size_t>(3 * minimum))
        reason |= 4;
    if (previous > live && previous - live >= static_cast<size_t>(minimum / 2) &&
        live < .75 * previous)
        reason |= 8;
    if (cells < 8)
        reason |= 16;
    return reason;
}
std::vector<cv::Point2f> cameraPixels(const std::vector<StereoObservation>& observations,
                                      int camera) {
    std::vector<cv::Point2f> pixels;
    for (const auto& observation : observations)
        if (observation.valid[camera])
            pixels.push_back(observation.pixel[camera]);
    return pixels;
}
int stereoCoverage(const std::vector<StereoObservation>& observations, cv::Size size) {
    return std::max(landmarkCoverage(cameraPixels(observations, 0), size),
                    landmarkCoverage(cameraPixels(observations, 1), size));
}
}
