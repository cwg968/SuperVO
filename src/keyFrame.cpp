#include <superPoint.h>
#include <keyFrame.h>
#include <stereoMatching.h>
#include <cmath>
#include <stdexcept>
namespace superVO {
bool createKeyframe(KeyframeData& frame, const StereoImages& images, SuperPointDetector& detector,
                    const cv::Mat& K, double baseline, const cv::Matx44d& pose, int minPoints,
                    int maxPoints, float threshold, int frameIndex, bool retainFeatures) {
    frame.stereoStats = {};
    const auto left = detector.extract(images.left, threshold, maxPoints);
    const auto right = detector.extract(images.right, threshold, maxPoints);
    const auto matches = matchSuperPointDescriptors(left.descriptors, right.descriptors);
    std::shared_ptr<KeyframeFeatures> features;
    if (retainFeatures) {
        features = std::make_shared<KeyframeFeatures>();
        features->pixels = left.pixels;
        features->descriptors = left.descriptors.clone();
        features->points.resize(left.pixels.size());
        features->rightPixels.resize(left.pixels.size());
        features->depthValid.resize(left.pixels.size(), 0);
    }
    frame.stereoStats.left = int(left.pixels.size());
    frame.stereoStats.right = int(right.pixels.size());
    frame.stereoStats.matches = int(matches.size());
    std::vector<StereoObservation> newObservations;
    std::vector<cv::Point3f> newPoints;
    double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2),
           cy = K.at<double>(1, 2);
    for (const auto& match : matches) {
        auto a = left.pixels[match.queryIdx], b = right.pixels[match.trainIdx];
        double disparity = a.x - b.x;
        if (disparity <= .5 || std::abs(a.y - b.y) > 1.0)
            continue;
        const auto refined = refineEpipolarStereo(images.left, images.right, a, b);
        if (refined.valid) {
            b = refined.right;
            disparity = a.x - b.x;
            ++frame.stereoStats.refined;
        } else {
            ++frame.stereoStats.refinementRejected;
            ++frame.stereoStats.refinementFallback;
        }
        double z = fx * baseline / disparity;
        if (!std::isfinite(z) || z < .2 || z > 80)
            continue;
        ++frame.stereoStats.geometry;
        if (features) {
            features->points.at(match.queryIdx) = {(a.x - cx) * z / fx, (a.y - cy) * z / fy, z};
            features->rightPixels.at(match.queryIdx) = b;
            features->depthValid.at(match.queryIdx) = 1;
        }
        bool duplicate = false;
        for (const auto& p : newObservations)
            if (cv::norm(p.pixel[0] - a) < 8) {
                duplicate = true;
                break;
            }
        if (duplicate)
            continue;
        if (newPoints.size() >= static_cast<size_t>(maxPoints))
            break;
        newObservations.push_back({{a, b}, {true, true}});
        newPoints.emplace_back((a.x - cx) * z / fx, (a.y - cy) * z / fy, z);
    }
    if (newPoints.size() < static_cast<size_t>(minPoints))
        return false;
    frame.observations = std::move(newObservations);
    frame.points = std::move(newPoints);
    frame.T_world_camera = pose;
    frame.birthFrames.assign(frame.points.size(), frameIndex);
    frame.ids.assign(frame.points.size(), 0);
    frame.retained = 0;
    frame.added = int(frame.points.size());
    frame.features = std::move(features);
    return true;
}
void addTrackedReserve(KeyframeData& frame,
                       const std::vector<StereoObservation>& trackedObservations,
                       const std::vector<cv::Point3f>& trackedPoints,
                       const std::vector<int>& births, const std::vector<LandmarkId>& trackedIds,
                       const cv::Matx44d& transform, const cv::Mat& K,
                       const cv::Matx44d& rightFromLeft, int frameIndex, int maxAge,
                       int maxPoints) {
    if (trackedObservations.size() != trackedPoints.size() ||
        births.size() != trackedPoints.size() || trackedIds.size() != trackedPoints.size() ||
        frame.ids.size() != frame.points.size() ||
        frame.observations.size() != frame.points.size() ||
        frame.points.size() != frame.birthFrames.size() || maxAge < 0 || maxPoints < 1)
        throw std::invalid_argument("Invalid landmark reserve");
    std::vector<StereoObservation> merged;
    std::vector<cv::Point3f> mergedPoints;
    std::vector<int> mergedBirths;
    std::vector<LandmarkId> mergedIds;
    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2),
                 cy = K.at<double>(1, 2);
    for (size_t i = 0; i < trackedPoints.size(); ++i) {
        if (frameIndex < births[i] || frameIndex - births[i] > maxAge)
            continue;
        const auto& p = trackedPoints[i];
        const cv::Vec4d q = transform * cv::Vec4d(p.x, p.y, p.z, 1);
        if (!std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2]) || q[2] < .2 ||
            q[2] > 80)
            continue;
        StereoObservation observation = trackedObservations[i];
        for (int c = 0; c < 2; ++c) {
            const auto qc = c ? rightFromLeft * q : q;
            const cv::Point2f projected(fx * qc[0] / qc[2] + cx, fy * qc[1] / qc[2] + cy);
            const auto pixel = observation.pixel[c];
            observation.valid[c] = observation.valid[c] && qc[2] > 0 && std::isfinite(pixel.x) &&
                                   std::isfinite(pixel.y) && cv::norm(projected - pixel) <= 1.5;
            // Only a location for deduplication when missing; valid=false means
            // it cannot become an LK seed or a PnP observation.
            if (!observation.valid[c])
                observation.pixel[c] = projected;
        }
        if (!observation.valid[0] && !observation.valid[1])
            continue;
        bool duplicate = false;
        for (const auto& other : merged)
            if (cv::norm(other.pixel[0] - observation.pixel[0]) < 8) {
                duplicate = true;
                break;
            }
        if (duplicate)
            continue;
        if (mergedPoints.size() >= static_cast<size_t>(maxPoints))
            break;
        merged.push_back(observation);
        mergedPoints.emplace_back(q[0], q[1], q[2]);
        mergedBirths.push_back(births[i]);
        mergedIds.push_back(trackedIds[i]);
    }
    frame.retained = int(mergedPoints.size());
    frame.added = 0;
    for (size_t i = 0; i < frame.points.size(); ++i) {
        bool duplicate = false;
        for (const auto& other : merged)
            if (cv::norm(other.pixel[0] - frame.observations[i].pixel[0]) < 8) {
                duplicate = true;
                break;
            }
        if (duplicate)
            continue;
        if (mergedPoints.size() >= static_cast<size_t>(maxPoints))
            break;
        merged.push_back(frame.observations[i]);
        mergedPoints.push_back(frame.points[i]);
        mergedBirths.push_back(frame.birthFrames[i]);
        mergedIds.push_back(frame.ids[i]);
        ++frame.added;
    }
    frame.observations = std::move(merged);
    frame.points = std::move(mergedPoints);
    frame.birthFrames = std::move(mergedBirths);
    frame.ids = std::move(mergedIds);
}
void assignLandmarkIds(KeyframeData& frame, LandmarkId& nextId) {
    if (frame.ids.size() != frame.points.size())
        throw std::invalid_argument("Misaligned landmark identities");
    for (auto& id : frame.ids)
        if (id == 0)
            id = nextId++;
}
}
