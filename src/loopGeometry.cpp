#include <loopGeometry.h>
#include <poseEstimation.h>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace superVO {
namespace {
cv::Matx33d rotation(const cv::Matx44d& t) {
    return {t(0, 0), t(0, 1), t(0, 2), t(1, 0), t(1, 1), t(1, 2), t(2, 0), t(2, 1), t(2, 2)};
}
void validate(const KeyframeFeatures& f) {
    if (f.descriptors.type() != CV_32F || f.descriptors.cols != 256 ||
        f.descriptors.rows != int(f.pixels.size()) || f.points.size() != f.pixels.size() ||
        f.depthValid.size() != f.pixels.size() || f.rightPixels.size() != f.pixels.size() ||
        !cv::checkRange(f.descriptors))
        throw std::invalid_argument("Invalid loop feature rows");
}
}
cv::Mat pooledSuperPoint(const cv::Mat& d) {
    if (d.empty() || d.type() != CV_32F || d.cols != 256 || !cv::checkRange(d))
        throw std::invalid_argument("Invalid aggregation input");
    cv::Mat pooled;
    cv::reduce(d, pooled, 0, cv::REDUCE_AVG, CV_32F);
    double n = cv::norm(pooled);
    if (n < 1e-8)
        throw std::invalid_argument("Degenerate global descriptor");
    pooled /= n;
    return pooled;
}
LoopVerification verifyLoop(const LoopKeyframe& reference, const LoopKeyframe& current,
                            const StereoCalibration& calibration, const LoopOptions& options) {
    LoopVerification out;
    if (!reference.features || !current.features)
        return out;
    const auto& a = *reference.features;
    const auto& b = *current.features;
    validate(a);
    validate(b);
    if (a.descriptors.rows < 2 || b.descriptors.rows < 2)
        return out;
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> ab, ba;
    matcher.knnMatch(a.descriptors, b.descriptors, ab, 2);
    matcher.knnMatch(b.descriptors, a.descriptors, ba, 2);
    std::vector<cv::Point3f> points, reversePoints;
    std::vector<cv::Point2f> pixels, reversePixels;
    std::array<PointTracks, 2> tracks;
    for (size_t i = 0; i < ab.size(); ++i) {
        if (ab[i].size() < 2)
            continue;
        auto m = ab[i][0];
        const auto& back = ba.at(m.trainIdx);
        if (m.distance > .7 || m.distance >= .8 * ab[i][1].distance || back.size() < 2 ||
            back[0].trainIdx != int(i) || back[0].distance >= .8 * back[1].distance ||
            !a.depthValid[i])
            continue;
        const auto& p = a.points[i];
        if (p.z < .3 || p.z > 40)
            continue;
        size_t j = m.trainIdx, id = points.size();
        points.push_back(p);
        pixels.push_back(b.pixels[j]);
        tracks[0].pixels.push_back(b.pixels[j]);
        tracks[0].sourceIndices.push_back(id);
        if (b.depthValid[j]) {
            tracks[1].pixels.push_back(b.rightPixels[j]);
            tracks[1].sourceIndices.push_back(id);
            reversePoints.push_back(b.points[j]);
            reversePixels.push_back(a.pixels[i]);
        }
    }
    out.matches = int(points.size());
    if (out.matches < options.minInliers || reversePoints.size() < size_t(options.minInliers))
        return out;
    auto forward = estimatePose(points, pixels, calibration.intrinsics, options.minInliers,
                                options.pixelThreshold);
    auto backward = estimatePose(reversePoints, reversePixels, calibration.intrinsics,
                                 options.minInliers, options.pixelThreshold);
    out.rejection = 2;
    if (!forward.valid || !backward.valid)
        return out;
    const auto cycle = forward.T_camera_reference * backward.T_camera_reference;
    cv::Vec3d angle;
    cv::Rodrigues(rotation(cycle), angle);
    if (cv::norm(angle) > .06 || cv::norm(cv::Vec3d(cycle(0, 3), cycle(1, 3), cycle(2, 3))) > .3)
        return out;
    auto stereo = estimateStereoPose(
        points, tracks, calibration.intrinsics, calibration.rightFromLeft(), calibration.imageSize,
        forward.T_camera_reference, options.minInliers, options.pixelThreshold);
    out.rejection = 3;
    if (!stereo.pose.valid)
        return out;
    out.inliers = int(stereo.quality.uniqueInliers);
    out.rmse = stereo.quality.rmse;
    out.coverage = stereo.quality.coverage;
    auto gr = rotation(reference.odomPose).t() * reference.gravity;
    auto gc = rotation(current.odomPose).t() * current.gravity;
    if (cv::norm(gr) < 1 || cv::norm(gc) < 1)
        return out;
    auto predicted = rotation(stereo.pose.T_camera_reference) * gr;
    out.tilt =
        std::acos(std::clamp(predicted.dot(gc) / (cv::norm(predicted) * cv::norm(gc)), -1., 1.));
    out.rejection = 4;
    if (out.tilt > .10 || out.coverage < .25 || out.inliers < options.minInliers ||
        out.inliers < .4 * out.matches)
        return out;
    out.valid = true;
    out.rejection = 0;
    out.currentFromReference = stereo.pose.T_camera_reference;
    return out;
}
}
