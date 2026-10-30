#include <templateMatcher.h>
#include <visualFrontend.h>
#include <opencv2/calib3d.hpp>
#include <chrono>
namespace superVO {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
PointTracks trackCamera(const cv::Mat& previous, const cv::Mat& current,
                        const std::vector<StereoObservation>& observations, int camera,
                        bool exposure = false) {
    std::vector<cv::Point2f> pixels;
    std::vector<size_t> indices;
    for (size_t j = 0; j < observations.size(); ++j)
        if (observations[j].valid[camera]) {
            pixels.push_back(observations[j].pixel[camera]);
            indices.push_back(j);
        }
    auto tracks = trackPoints(previous, current, pixels, exposure);
    for (auto& j : tracks.sourceIndices)
        j = indices.at(j);
    return tracks;
}
PointTracks guideCamera(const cv::Mat& previous, const cv::Mat& current,
                        const std::vector<StereoObservation>& observations,
                        const std::vector<cv::Point3f>& points, const cv::Matx44d& transform,
                        const cv::Mat& K, int camera) {
    const auto projections = projectLandmarks(points, transform, K, current.size());
    std::vector<cv::Point2f> pixels, predictions;
    std::vector<size_t> indices;
    for (size_t j = 0; j < observations.size(); ++j)
        if (observations[j].valid[camera]) {
            pixels.push_back(observations[j].pixel[camera]);
            predictions.push_back(projections[j]);
            indices.push_back(j);
        }
    auto tracks = trackGuidedPoints(previous, current, pixels, predictions);
    for (auto& j : tracks.sourceIndices)
        j = indices.at(j);
    return tracks;
}
PoseEstimate cameraEstimate(const std::vector<cv::Point3f>& points, const PointTracks& tracks,
                            const cv::Mat& K, const cv::Matx44d& rightFromLeft, int camera,
                            int minimum) {
    std::vector<cv::Point3f> tracked;
    for (auto j : tracks.sourceIndices)
        tracked.push_back(points.at(j));
    auto estimate = estimatePose(tracked, tracks.pixels, K, minimum);
    estimate.valid = estimate.valid && estimate.inliers.size() >= .4 * tracked.size();
    if (camera && estimate.valid)
        estimate.T_camera_reference = leftPoseFromRight(estimate.T_camera_reference, rightFromLeft);
    return estimate;
}
StereoPoseEstimate recoverStereo(const std::vector<cv::Point3f>& points,
                                 const std::array<PointTracks, 2>& primary,
                                 const std::array<PointTracks, 2>& guided, const cv::Mat& K,
                                 const cv::Matx44d& E, cv::Size size, const cv::Matx44d& initial,
                                 int minimum, std::array<PointTracks, 2>& accepted, int& attempts,
                                 int& observationSet) {
    StereoPoseEstimate best;
    for (int variant = 0; variant < 4; ++variant) {
        auto observations = primary;
        bool distinct = variant == 0;
        for (int c = 0; c < 2; ++c)
            if ((variant & (1 << c)) && !guided[c].pixels.empty()) {
                distinct |= guided[c].pixels != primary[c].pixels ||
                            guided[c].sourceIndices != primary[c].sourceIndices;
                observations[c] = guided[c];
            }
        if (!distinct)
            continue;
        ++attempts;
        auto result = estimateStereoPose(points, observations, K, E, size, initial, minimum);
        if (result.pose.valid && (!best.pose.valid || result.quality.score > best.quality.score)) {
            best = std::move(result);
            accepted = std::move(observations);
            observationSet = variant;
        } else if (!best.pose.valid && result.uniqueObservations >= best.uniqueObservations)
            best = std::move(result);
    }
    return best;
}
}
VisualTrackingResult trackVisualFrame(const StereoImages& previous, const StereoImages& images,
                                      const TrackingReference& active, const LocalMap& localMap,
                                      const cv::Matx44d& prediction,
                                      const StereoCalibration& calibration,
                                      const TrackingOptions& o, int i) {
    VisualTrackingResult result;
    result.reference = active;
    auto& reference = result.reference.geometry;
    auto& points = reference.points;
    auto& observations = reference.observations;
    auto& births = reference.birthFrames;
    auto& ids = reference.ids;
    auto& lastKeyframe = result.reference.frameIndex;
    auto& anchorCount = result.reference.anchorCount;
    auto& d = result.diagnostics;
    d.landmarksBefore = points.size();
    const auto rightFromLeft = calibration.rightFromLeft();
    for (int c = 0; c < 2; ++c) {
        const auto& before = c ? previous.right : previous.left;
        const auto& current = c ? images.right : images.left;
        auto t = Clock::now();
        d.tracks[c] = trackCamera(before, current, observations, c);
        ++d.lkAttempts;
        d.trackTime += ms(t);
        d.rawCounts[c] = d.tracks[c].pixels.size();
        t = Clock::now();
        d.candidates[c] = cameraEstimate(points, d.tracks[c], calibration.intrinsics, rightFromLeft,
                                         c, o.minInliers);
        d.poseTime += ms(t);
        if (!d.candidates[c].valid) {
            d.retries[c] = true;
            t = Clock::now();
            auto retry = trackCamera(before, current, observations, c, true);
            ++d.lkAttempts;
            d.trackTime += ms(t);
            if (retry.exposureCompensated) {
                d.gains[c] = retry.exposureGain;
                d.biases[c] = retry.exposureBias;
                t = Clock::now();
                auto recovered = cameraEstimate(points, retry, calibration.intrinsics,
                                                rightFromLeft, c, o.minInliers);
                d.poseTime += ms(t);
                if (recovered.valid) {
                    d.tracks[c] = std::move(retry);
                    d.candidates[c] = std::move(recovered);
                    d.recoveries[c] = true;
                }
            }
        }
    }
    const auto predictedTransform = prediction.inv(cv::DECOMP_SVD) * reference.T_world_camera;
    for (int c = 0; c < 2; ++c)
        if (!d.candidates[c].valid) {
            d.guidedAttempted[c] = true;
            auto t = Clock::now();
            const auto transform = c ? rightFromLeft * predictedTransform : predictedTransform;
            auto recoveredTracks =
                guideCamera(c ? previous.right : previous.left, c ? images.right : images.left,
                            observations, points, transform, calibration.intrinsics, c);
            ++d.lkAttempts;
            d.trackTime += ms(t);
            d.guidedCounts[c] = recoveredTracks.pixels.size();
            t = Clock::now();
            auto recovered = cameraEstimate(points, recoveredTracks, calibration.intrinsics,
                                            rightFromLeft, c, o.minInliers);
            d.poseTime += ms(t);
            d.guidedTracks[c] = recoveredTracks;
            if (recovered.valid) {
                d.tracks[c] = std::move(recoveredTracks);
                d.candidates[c] = std::move(recovered);
                d.guidedRecovered[c] = true;
            }
        }
    if (!d.candidates[0].valid && !d.candidates[1].valid) {
        auto t = Clock::now();
        std::array<PointTracks, 2> jointTracks;
        d.jointEstimate =
            recoverStereo(points, d.tracks, d.guidedTracks, calibration.intrinsics, rightFromLeft,
                          calibration.imageSize, predictedTransform, o.minInliers, jointTracks,
                          d.stereoAttempts, d.stereoObservationSet);
        d.poseTime += ms(t);
        if (d.jointEstimate.pose.valid)
            d.tracks = std::move(jointTracks);
    }
    if (!d.candidates[0].valid && !d.candidates[1].valid && !d.jointEstimate.pose.valid) {
        size_t mapTrackingCalls = 0;
        auto t = Clock::now();
        auto recoveredMaps =
            TemplateMatcher::match(localMap, images, prediction, calibration.intrinsics,
                                   rightFromLeft, i, &mapTrackingCalls);
        d.trackTime += ms(t);
        d.lkAttempts += mapTrackingCalls;
        for (auto& map : recoveredMaps) {
            ++d.mapCandidates;
            t = Clock::now();
            std::array<PoseEstimate, 2> estimates;
            std::array<StereoPoseQuality, 2> assessments;
            for (int c = 0; c < 2; ++c) {
                estimates[c] = cameraEstimate(map.points, map.tracks[c], calibration.intrinsics,
                                              rightFromLeft, c, o.minInliers);
                if (estimates[c].valid)
                    assessments[c] = assessStereoPose(estimates[c].T_camera_reference, map.points,
                                                      map.tracks, calibration.intrinsics,
                                                      rightFromLeft, calibration.imageSize);
            }
            int choice = selectStereoPose(estimates, assessments, o.minInliers);
            StereoPoseEstimate mapJoint;
            if (choice < 0) {
                ++d.stereoAttempts;
                mapJoint = estimateStereoPose(
                    map.points, map.tracks, calibration.intrinsics, rightFromLeft,
                    calibration.imageSize, prediction.inv(cv::DECOMP_SVD) * map.worldFromReference,
                    o.minInliers);
                if (mapJoint.pose.valid)
                    choice = 2;
            }
            d.poseTime += ms(t);
            if (choice < 0 || (choice < 2 && assessments[choice].coverage < .25))
                continue;
            // Only a verified image match can replace the active chain.
            d.mapRecovered = true;
            d.mapSource = map.sourceFrame;
            points = std::move(map.points);
            observations = std::move(map.observations);
            births = std::move(map.births);
            ids = std::move(map.ids);
            reference.T_world_camera = map.worldFromReference;
            lastKeyframe = map.sourceFrame;
            anchorCount = points.size();
            d.candidates = std::move(estimates);
            d.tracks = std::move(map.tracks);
            if (choice == 2) {
                d.jointEstimate = std::move(mapJoint);
                d.stereoObservationSet = 4;
            }
            break;
        }
    }
    auto t = Clock::now();
    for (int c = 0; c < 2; ++c)
        if (d.candidates[c].valid)
            d.quality[c] =
                assessStereoPose(d.candidates[c].T_camera_reference, points, d.tracks,
                                 calibration.intrinsics, rightFromLeft, calibration.imageSize);
    d.selected = static_cast<PoseSource>(selectStereoPose(d.candidates, d.quality, o.minInliers));
    if (d.selected == PoseSource::None && d.jointEstimate.pose.valid)
        d.selected = PoseSource::Stereo;
    d.poseTime += ms(t);
    d.valid = d.selected != PoseSource::None;
    if (d.candidates[0].valid && d.candidates[1].valid) {
        const auto delta = d.candidates[0].T_camera_reference *
                           d.candidates[1].T_camera_reference.inv(cv::DECOMP_SVD);
        d.disagreementM = cv::norm(cv::Vec3d(delta(0, 3), delta(1, 3), delta(2, 3)));
        cv::Matx33d R(delta(0, 0), delta(0, 1), delta(0, 2), delta(1, 0), delta(1, 1), delta(1, 2),
                      delta(2, 0), delta(2, 1), delta(2, 2));
        cv::Vec3d r;
        cv::Rodrigues(R, r);
        d.disagreementDeg = cv::norm(r) * 180 / CV_PI;
    }
    const int logged =
        d.valid && d.selected != PoseSource::Stereo ? static_cast<int>(d.selected) : 0;
    d.count = d.tracks[logged].pixels.size();
    d.inliers = d.candidates[logged].inliers.size();
    d.rawCount = d.rawCounts[logged];
    if (d.selected == PoseSource::Stereo) {
        d.count = d.jointEstimate.uniqueObservations;
        d.inliers = d.jointEstimate.pose.inliers.size();
    }
    d.exposureRetry = d.retries[logged];
    d.exposureRecovered = d.recoveries[logged];
    d.exposureGain = d.gains[logged];
    d.exposureBias = d.biases[logged];
    if (d.valid) {
        VisualPoseCandidate candidate;
        candidate.source = d.selected;
        candidate.referenceFrame = result.reference.frameIndex;
        candidate.T_world_reference = reference.T_world_camera;
        const int camera = static_cast<int>(d.selected);
        candidate.T_current_reference = d.selected == PoseSource::Stereo
                                            ? d.jointEstimate.pose.T_camera_reference
                                            : d.candidates[camera].T_camera_reference;
        candidate.quality =
            d.selected == PoseSource::Stereo ? d.jointEstimate.quality : d.quality[camera];
        std::vector<bool> supported(points.size(), false);
        for (int c = 0; c < 2; ++c)
            for (auto j : candidate.quality.inlierObservations[c])
                supported.at(d.tracks[c].sourceIndices.at(j)) = true;
        for (size_t j = 0; j < supported.size(); ++j)
            if (supported[j]) {
                candidate.landmarkIndices.push_back(j);
                candidate.landmarkIds.push_back(ids.at(j));
            }
        result.candidate = std::move(candidate);
    }
    return result;
}
}
