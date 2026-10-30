#include <odometry.h>
#include <trackingPolicy.h>
#include <localMap.h>
#include <visualFrontend.h>
#include <motionPrediction.h>
#include <inertialEstimator.h>
#include <vioEstimator.h>
#include <recoveryPolicy.h>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <set>
#include <map>
namespace superVO {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}
struct TrackingState {
    TrackingReference active;
    LocalMap localMap;
    LandmarkId nextId = 1;
    StereoImages previous;
    int lastAttempt = -2;
    ConstantVelocityPredictor predictor;
    int frameIndex = 0;
    TimestampNs timestampNs = 0;
    std::optional<VioEstimator> filter;
#ifdef SUPERVO_WITH_BA
    std::optional<StereoBaEstimator> visualBa;
#endif
    std::optional<InertialInitializer> initializer;
    TimestampNs firstTimestamp = 0;
    RecoveryPolicy recovery;
    StationaryDetector stationary;
    TimestampNs lastStationaryUpdate = -1;
    int reliableReference = -1;
    bool currentMeasurements = false;
    explicit TrackingState(int interval, std::optional<ImuCalibration> imu,
                           const StereoCalibration& camera, VioOptions vio, int minimum,
                           StereoBaOptions ba)
        : localMap(5, 2 * interval) {
        if (ba.enabled) {
            if (imu)
                throw std::invalid_argument(
                    "Pure stereo BA and IMU estimators are mutually exclusive");
#ifdef SUPERVO_WITH_BA
            ba.minInliers = minimum;
            visualBa.emplace(camera, ba);
#else
            throw std::invalid_argument("Visual BA requires SUPERVO_WITH_BA=ON");
#endif
        }
        if (imu) {
            vio.window.minInliers = minimum;
            filter.emplace(camera, *imu, vio);
            initializer.emplace(*imu);
        }
    }
};
struct Odometry::Impl {
    StereoCalibration calibration;
    TrackingOptions o;
    VioOptions vio;
    std::optional<ImuCalibration> imuCalibration;
    SuperPointDetector& detector;
    TrackingState state;
    bool stopped = false;
    Impl(StereoCalibration c, TrackingOptions options, SuperPointDetector& featureDetector,
         std::optional<ImuCalibration> imu, VioOptions backendOptions, StereoBaOptions ba)
        : calibration(std::move(c)), o(options), vio(backendOptions), imuCalibration(imu),
          detector(featureDetector), state(options.keyframeInterval, imu, calibration,
                                           backendOptions, options.minInliers, ba) {
        calibration.validate();
        if (vio.backend == VioBackend::Window && !imu)
            throw std::invalid_argument("Sliding-window VIO requires calibrated IMU input");
        if (o.minInliers < 6 || o.maxPoints < o.minInliers || o.keyframeInterval < 1 ||
            !std::isfinite(o.threshold) || o.threshold <= 0 || o.threshold >= 1)
            throw std::invalid_argument("Invalid tracking configuration");
        calibration.intrinsics = calibration.intrinsics.clone();
    }
    TrackingResult process(const StereoFrame& frame, const std::vector<ImuSample>& imu);
};
Odometry::Odometry(StereoCalibration c, TrackingOptions o, SuperPointDetector& detector,
                   std::optional<ImuCalibration> imu, VioOptions vio, StereoBaOptions ba)
    : impl_(std::make_unique<Impl>(std::move(c), o, detector, imu, vio, ba)) {}
Odometry::~Odometry() = default;
OdometrySnapshot Odometry::snapshot() const {
    const auto& s = impl_->state;
    return {s.frameIndex, s.timestampNs, s.predictor.lastPose(), s.active.geometry.ids};
}
VisualizationSnapshot Odometry::visualization() const {
    const auto& s = impl_->state;
    const auto& reference = s.active.geometry;
    VisualizationSnapshot result;
    result.timestampNs = s.timestampNs;
    if (!s.frameIndex)
        return result;
    if (s.currentMeasurements) {
        result.measuredLeft = cameraPixels(reference.observations, 0);
        if (s.active.frameIndex == s.frameIndex - 1 && reference.features)
            result.extractedLeft = reference.features->pixels;
    }
    std::map<LandmarkId, cv::Point3d> points;
    for (size_t k = 0; k < reference.points.size(); ++k) {
        const auto& p = reference.points[k];
        const auto q = reference.T_world_camera * cv::Vec4d(p.x, p.y, p.z, 1);
        points.emplace(reference.ids.at(k), cv::Point3d(q[0], q[1], q[2]));
    }
    if (s.filter && s.filter->initialized() && s.filter->usesPixels())
        for (const auto& point : s.filter->optimizedLandmarks())
            points[point.landmark] = point.world;
#ifdef SUPERVO_WITH_BA
    if (s.visualBa)
        for (const auto& point : s.visualBa->landmarks())
            points[point.landmark] = point.world;
#endif
    for (const auto& [id, point] : points)
        if (id && std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
            result.landmarks.push_back({id, point});
    return result;
}
TrackingResult Odometry::process(const StereoFrame& frame, const std::vector<ImuSample>& imu) {
    return impl_->process(frame, imu);
}
TrackingResult Odometry::Impl::process(const StereoFrame& frame,
                                       const std::vector<ImuSample>& imu) {
    if (stopped)
        throw std::logic_error("Odometry is stopped after tracking failure");
    if (frame.timestampNs < 0 || (state.frameIndex && frame.timestampNs <= state.timestampNs))
        throw std::invalid_argument("Frame timestamps must increase");
    if (frame.images.left.type() != CV_8UC1 || frame.images.right.type() != CV_8UC1 ||
        frame.images.left.size() != calibration.imageSize ||
        frame.images.right.size() != calibration.imageSize)
        throw std::invalid_argument("Invalid rectified stereo frame");
    // Work on a candidate state. Only the final successful result is committed.
    if (imuCalibration)
        validateImuInterval(imu, state.frameIndex ? state.timestampNs : frame.timestampNs,
                            frame.timestampNs, imuCalibration->rateHz);
    auto next = state;
    auto& reference = next.active.geometry;
    auto& localMap = next.localMap;
    auto& observations = reference.observations;
    auto& points = reference.points;
    auto& births = reference.birthFrames;
    auto& ids = reference.ids;
    auto& previous = next.previous;
    auto& anchorCount = next.active.anchorCount;
    auto& lastKeyframe = next.active.frameIndex;
    auto& lastAttempt = next.lastAttempt;

    const int i = next.frameIndex;
    const auto& images = frame.images;
    const auto rightFromLeft = calibration.rightFromLeft();
    TrackingResult result;
    result.timestampNs = frame.timestampNs;
    result.frameIndex = i;
    auto& d = result.diagnostics;
    d.landmarksBefore = points.size();
    reference.stereoStats = {};
    bool inertialReady = next.filter && next.filter->initialized();
    if (i == 0)
        next.firstTimestamp = frame.timestampNs;
    if (inertialReady) {
        auto timer = Clock::now();
        next.filter->propagate(imu);
        if (auto updated = next.filter->referencePose(lastKeyframe))
            reference.T_world_camera = *updated;
        for (int source : localMap.referenceFrames())
            if (auto updated = next.filter->referencePose(source))
                localMap.updateReferencePose(source, *updated);
        if (next.filter->usesPixels()) {
            const auto optimized = next.filter->optimizedLandmarks();
            updateLandmarkGeometry(reference.points, reference.ids, reference.T_world_camera,
                                   optimized);
            localMap.updateLandmarks(optimized);
        }
        result.inertialMs += ms(timer);
    }
    VisualMeasurements baMeasurements;
    cv::Matx44d pose = cv::Matx44d::eye();
    if (i == 0) {
        auto t = Clock::now();
        d.attempted = true;
        d.valid = createKeyframe(reference, images, detector, calibration.intrinsics,
                                 calibration.baseline, pose, o.minInliers, o.maxPoints, o.threshold,
                                 i, o.retainFeatures);
        d.keyTime = ms(t);
        if (d.valid) {
            assignLandmarkIds(reference, next.nextId);
            anchorCount = points.size();
            d.key = true;
            d.inliers = points.size();
            d.jointInliers = points.size();
            d.stereoLandmarks = int(points.size());
            d.added = d.stereoLandmarks;
        }
    } else {
        const auto prediction = inertialReady ? next.filter->state().T_world_rect
                                              : next.predictor.predict(frame.timestampNs);
        auto visual = trackVisualFrame(previous, images, next.active, localMap, prediction,
                                       calibration, o, i);
        d = std::move(visual.diagnostics);
        if (visual.candidate && inertialReady) {
            auto timer = Clock::now();
            VisualMeasurements measurements;
            if (next.filter->usesPixels()) {
                const auto& geometry = visual.reference.geometry;
                std::vector<StereoObservation> pixels(geometry.points.size());
                for (int camera = 0; camera < 2; ++camera)
                    for (size_t k = 0; k < d.tracks[camera].pixels.size(); ++k) {
                        const auto index = d.tracks[camera].sourceIndices.at(k);
                        pixels.at(index).pixel[camera] = d.tracks[camera].pixels[k];
                        pixels.at(index).valid[camera] = true;
                    }
                measurements =
                    measuredPixels(images, i, frame.timestampNs, geometry.ids, geometry.points,
                                   pixels, geometry.T_world_camera, vio.noise);
            }
            result.inertialUpdate = next.filter->correct(
                visual.candidate->T_current_reference, visual.candidate->referenceFrame,
                visual.candidate->quality.rmse, visual.candidate->quality.uniqueInliers,
                measurements);
            result.inertialMs += ms(timer);
            if (!result.inertialUpdate) {
                visual.candidate.reset();
                d.valid = false;
            } else if (auto updated = next.filter->referencePose(visual.reference.frameIndex))
                visual.reference.geometry.T_world_camera = *updated;
            if (result.inertialUpdate && next.filter->usesPixels())
                updateLandmarkGeometry(
                    visual.reference.geometry.points, visual.reference.geometry.ids,
                    visual.reference.geometry.T_world_camera, next.filter->optimizedLandmarks());
        }
        if (visual.candidate)
            next.active = std::move(visual.reference);
        if (d.valid) {
            auto transform = visual.candidate->T_current_reference;
            const auto& assessment = visual.candidate->quality;
            pose = inertialReady ? next.filter->state().T_world_rect
                                 : cameraPose(reference.T_world_camera, transform);
            if (inertialReady)
                transform = pose.inv(cv::DECOMP_SVD) * reference.T_world_camera;
            d.jointInliers = assessment.uniqueInliers;
            std::vector<StereoObservation> observed(points.size());
            for (int c = 0; c < 2; ++c)
                for (auto j : assessment.inlierObservations[c]) {
                    auto& observation = observed.at(d.tracks[c].sourceIndices.at(j));
                    observation.pixel[c] = d.tracks[c].pixels.at(j);
                    observation.valid[c] = true;
                }
            if (inertialReady && next.filter->usesPixels())
                discardPosteriorOutliers(i, ids, observed, next.filter->feedback());
#ifdef SUPERVO_WITH_BA
            if (next.visualBa)
                baMeasurements =
                    measuredPixels(images, i, frame.timestampNs, ids, points, observed,
                                   reference.T_world_camera, ObservationNoise::Uniform);
#endif
            const int oldLeftCount = int(cameraPixels(observations, 0).size()),
                      oldRightCount = int(cameraPixels(observations, 1).size());
            std::vector<cv::Point3f> keptPoints;
            std::vector<int> keptBirths;
            std::vector<LandmarkId> keptIds;
            observations.clear();
            for (size_t j = 0; j < points.size(); ++j)
                if (observed[j].valid[0] || observed[j].valid[1]) {
                    observations.push_back(observed[j]);
                    keptPoints.push_back(points[j]);
                    keptBirths.push_back(births[j]);
                    keptIds.push_back(ids[j]);
                }
            points = std::move(keptPoints);
            births = std::move(keptBirths);
            ids = std::move(keptIds);
            d.reason =
                replenishmentReason(i - lastKeyframe, points.size(), anchorCount, d.landmarksBefore,
                                    stereoCoverage(observations, calibration.imageSize),
                                    o.minInliers, o.keyframeInterval);
            const int leftCount = int(cameraPixels(observations, 0).size()),
                      rightCount = int(cameraPixels(observations, 1).size());
            if ((leftCount < o.minInliers &&
                 (oldLeftCount >= o.minInliers || i - lastAttempt >= o.keyframeInterval)) ||
                (rightCount < o.minInliers &&
                 (oldRightCount >= o.minInliers || i - lastAttempt >= o.keyframeInterval)))
                d.reason |= 32;
            if (d.reason &&
                (i - lastAttempt >= 2 || points.size() < static_cast<size_t>(2 * o.minInliers))) {
                auto t = Clock::now();
                d.attempted = true;
                lastAttempt = i;
                KeyframeData candidate;
                if (createKeyframe(candidate, images, detector, calibration.intrinsics,
                                   calibration.baseline, pose, 1, o.maxPoints, o.threshold, i,
                                   o.retainFeatures)) {
                    d.stereoLandmarks = int(candidate.points.size());
                    const int oldCells = stereoCoverage(observations, calibration.imageSize);
                    if (candidate.points.size() <
                            std::max(points.size(), static_cast<size_t>(3 * o.minInliers)) ||
                        stereoCoverage(candidate.observations, calibration.imageSize) < oldCells)
                        addTrackedReserve(candidate, observations, points, births, ids, transform,
                                          calibration.intrinsics, rightFromLeft, i,
                                          2 * o.keyframeInterval, o.maxPoints);
                    d.retained = candidate.retained;
                    d.added = candidate.added;
                    d.key =
                        candidate.points.size() >=
                            std::max(points.size(), static_cast<size_t>(o.minInliers)) &&
                        stereoCoverage(candidate.observations, calibration.imageSize) >= oldCells &&
                        d.added >= o.minInliers / 2;
                    if (d.key) {
                        assignLandmarkIds(candidate, next.nextId);
                        reference = std::move(candidate);
                        anchorCount = points.size();
                        lastKeyframe = i;
                    } else
                        reference.stereoStats = candidate.stereoStats;
                } else
                    reference.stereoStats = candidate.stereoStats;
                d.keyTime = ms(t);
            }
        }
        result.visual = std::move(visual.candidate);
    }
#ifdef SUPERVO_WITH_BA
    if (d.valid && next.visualBa) {
        auto timer = Clock::now();
        if (d.key) {
            auto fresh = measuredPixels(images, i, frame.timestampNs, ids, points, observations,
                                        reference.T_world_camera, ObservationNoise::Uniform);
            if (i == 0)
                baMeasurements = std::move(fresh);
            else {
                // One frame may observe both the outgoing and incoming point sets.
                // Preserve that bridge without counting a pixel/seed twice.
                std::set<std::pair<LandmarkId, int>> measured;
                std::set<LandmarkId> seeded;
                for (const auto& p : baMeasurements.pixels)
                    measured.emplace(p.landmark, p.camera);
                for (const auto& p : baMeasurements.seeds)
                    seeded.insert(p.landmark);
                for (const auto& p : fresh.pixels)
                    if (measured.emplace(p.landmark, p.camera).second)
                        baMeasurements.pixels.push_back(p);
                for (const auto& p : fresh.seeds)
                    if (seeded.insert(p.landmark).second)
                        baMeasurements.seeds.push_back(p);
            }
        }
        const auto initial = pose;
        if (next.visualBa->update(baMeasurements, pose, d.key)) {
            pose = *next.visualBa->pose(i);
            if (auto updated = next.visualBa->pose(lastKeyframe))
                reference.T_world_camera = *updated;
            for (int source : localMap.referenceFrames())
                if (auto updated = next.visualBa->pose(source))
                    localMap.updateReferencePose(source, *updated);
            const auto optimized = next.visualBa->landmarks();
            updateLandmarkGeometry(points, ids, reference.T_world_camera, optimized);
            localMap.updateLandmarks(optimized);
            next.predictor.rebase(pose * initial.inv(cv::DECOMP_SVD));
        }
        result.stereoBa = next.visualBa->diagnostics();
        result.baPoses = next.visualBa->poses();
        result.baReferenceFrame = lastKeyframe;
        result.stereoBaMs = ms(timer);
    }
#endif
    if (d.valid && next.filter)
        next.recovery.acceptedConstraint(frame.timestampNs, true);
    if (next.initializer && !inertialReady && d.valid) {
        auto timer = Clock::now();
        auto initial = next.initializer->observe(pose, frame.timestampNs, imu);
        if (initial) {
            next.filter->initialize(*initial, frame.timestampNs, i);
            next.initializer.reset();
            inertialReady = true;
            // Reanchor measured geometry at initialization; old templates have no correlated clones.
            const auto transform = pose.inv(cv::DECOMP_SVD) * reference.T_world_camera;
            for (auto& point : points) {
                auto q = transform * cv::Vec4d(point.x, point.y, point.z, 1);
                point = {float(q[0]), float(q[1]), float(q[2])};
            }
            reference.T_world_camera = pose;
            lastKeyframe = i;
            anchorCount = points.size();
            localMap = LocalMap(5, 2 * o.keyframeInterval);
            // Reanchoring geometry is not a new image extraction. The old
            // descriptor snapshot cannot be labelled with this frame's time.
            reference.features.reset();
            d.key = true;
        } else if ((frame.timestampNs - next.firstTimestamp) * 1e-9 > 10)
            d.valid = false;
        result.inertialMs += ms(timer);
    }
    if (next.filter) {
        auto timer = Clock::now();
        result.stationary =
            next.stationary.observe(frame, imu, next.filter->state(), *imuCalibration);
        result.inertialMs += ms(timer);
    }
    bool coasting = false;
    if (!d.valid && inertialReady) {
        if (result.stationary.stationary &&
            (next.lastStationaryUpdate < 0 ||
             frame.timestampNs - next.lastStationaryUpdate >= 300000000)) {
            auto timer = Clock::now();
            result.zeroVelocityUpdate = next.filter->correctStationary(
                result.stationary.meanAngularVelocity, result.stationary.meanAcceleration);
            if (result.zeroVelocityUpdate) {
                next.lastStationaryUpdate = frame.timestampNs;
                next.recovery.acceptedConstraint(frame.timestampNs, false);
            }
            result.inertialMs += ms(timer);
        }
        const auto inertial = next.filter->state();
        const auto uncertainty = next.filter->relativeUncertainty(next.reliableReference);
        coasting =
            next.recovery.mayPropagate(frame.timestampNs, uncertainty.first, uncertainty.second);
        if (coasting) {
            pose = inertial.T_world_rect;
            // Fresh stereo gives depth, not an independent pose; this frame remains PROPAGATING.
            auto timer = Clock::now();
            KeyframeData seed;
            if (createKeyframe(seed, images, detector, calibration.intrinsics, calibration.baseline,
                               pose, o.minInliers, o.maxPoints, o.threshold, i, o.retainFeatures)) {
                assignLandmarkIds(seed, next.nextId);
                reference = std::move(seed);
                lastKeyframe = i;
                anchorCount = points.size();
                d.key = true;
                result.inertialReseed = true;
            }
            d.keyTime += ms(timer);
        }
    }
    if ((d.valid || coasting) && d.key) {
        if (d.valid && inertialReady)
            result.keyframeFeatures = reference.features;
        auto t = Clock::now();
        localMap.remember(images, points, observations, births, pose, i, ids);
        if (inertialReady) {
            if (next.filter->usesPixels())
                next.filter->seed(measuredPixels(images, i, frame.timestampNs, ids, points,
                                                 observations, reference.T_world_camera,
                                                 vio.noise));
            next.filter->rememberReference(i);
        }
        d.keyTime += ms(t);
    }
    if (inertialReady && d.valid) {
        next.reliableReference = lastKeyframe;
        next.filter->protectReference(lastKeyframe);
        localMap.protectReference(lastKeyframe);
    }
    result.recoveryPhase = next.recovery.phase();
    result.sinceConstraintSeconds = next.recovery.sinceConstraint(frame.timestampNs);
    d.landmarksAfter = points.size();
    d.cells = stereoCoverage(observations, calibration.imageSize);
    d.mapSize = localMap.size();
    d.stereoStats = reference.stereoStats;
    result.status = d.valid  ? TrackingStatus::Tracking
                    : i == 0 ? TrackingStatus::InitializationFailed
                             : TrackingStatus::Lost;
    if (next.filter) {
        result.inertial = next.filter->state();
        result.backend = next.filter->diagnostics();
        if (d.valid && !inertialReady)
            result.status = TrackingStatus::InitializingInertial;
        if (coasting)
            result.status = result.zeroVelocityUpdate ? TrackingStatus::StationaryInertial
                                                      : TrackingStatus::PropagatingInertial;
    }
    if (d.valid || coasting) {
        result.T_world_rect = pose;
        next.predictor.accept(pose, frame.timestampNs);
        if (d.valid || d.key)
            previous = {images.left.clone(), images.right.clone()};
        next.timestampNs = frame.timestampNs;
        next.currentMeasurements = d.valid || d.key;
        ++next.frameIndex;
        state = std::move(next);
    } else
        stopped = true;
    return result;
}
}
