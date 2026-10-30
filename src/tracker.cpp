#include <superPoint.h>
#include <tracker.h>
#include <stereoDataset.h>
#include <odometry.h>
#include <imuDataset.h>
#include <frame.h>
#include <keyFrame.h>
#include <localMap.h>
#include <poseEstimation.h>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <array>
#include <map>
#ifdef SUPERVO_WITH_LOOP
#include <loopClosure.h>
#endif
#include <stdexcept>

namespace superVO {
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
void stamp(std::ostream& out, int64_t ns) {
    out << ns / 1000000000 << '.' << std::setw(9) << std::setfill('0') << ns % 1000000000
        << std::setfill(' ');
}
void writePose(std::ostream& out, int64_t timestamp, const cv::Matx44d& pose) {
    cv::Matx33d R(pose(0, 0), pose(0, 1), pose(0, 2), pose(1, 0), pose(1, 1), pose(1, 2),
                  pose(2, 0), pose(2, 1), pose(2, 2));
    cv::Vec3d axis;
    cv::Rodrigues(R, axis);
    double angle = cv::norm(axis);
    cv::Vec3d q = angle < 1e-12 ? cv::Vec3d(0, 0, 0) : axis * (std::sin(angle / 2) / angle);
    stamp(out, timestamp);
    out << std::setprecision(12) << ' ' << pose(0, 3) << ' ' << pose(1, 3) << ' ' << pose(2, 3)
        << ' ' << q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << std::cos(angle / 2) << '\n';
}
}
int runOdometry(const RunOptions& o) {
#ifndef SUPERVO_WITH_LOOP
    if (o.loop.enabled)
        throw std::invalid_argument("Loop closure requires SUPERVO_WITH_LOOP=ON");
#endif
    if (o.loop.enabled && !o.imu)
        throw std::invalid_argument("Gravity-aligned loop closure requires IMU");
    if (o.stereoBa.enabled && o.imu)
        throw std::invalid_argument("Pure stereo BA and IMU estimators are mutually exclusive");
    if (o.vio.window.auditEvery < 0 ||
        (o.vio.window.auditEvery > 0 && (!o.imu || o.vio.backend != VioBackend::Window)))
        throw std::invalid_argument(
            "Window audit requires window VIO and a nonnegative sampling interval");
    cv::setNumThreads(o.threads);
    StereoDataset dataset(o.dataset, o.format, o.imageScale, o.kittiCameras);
    const auto rightFromLeft = dataset.rightFromLeft();
    const auto calibration = dataset.calibration();
    std::optional<ImuDataset> imu;
    if (o.imu)
        imu.emplace(dataset.root(), calibration);
    std::cout << "format=" << dataset.format() << " pairs=" << dataset.samples().size()
              << " image=" << dataset.size() << " baseline_m=" << dataset.baseline()
              << " unpaired_left=" << dataset.unpairedLeft()
              << " unpaired_right=" << dataset.unpairedRight() << '\n';
    if (o.inspect) {
        auto images = dataset.read(0);
        return images.left.empty() ? 1 : 0;
    }
    std::filesystem::create_directories(o.output);
    for (const auto& name :
         {"trajectory.tum", "frames.csv", "run.yaml", "inertial.csv", "backend.csv",
          "stereo_ba.csv", "trajectory_ba_final.tum", "window_residuals.csv",
          "window_imu_weights.csv", "trajectory_map_online.tum", "trajectory_map_final.tum",
          "loop_keyframes", "loop_candidates.csv"})
        if (!o.overwrite && std::filesystem::exists(o.output / name))
            throw std::runtime_error("Output exists; choose another --output or use --overwrite");
    SuperPointDetector detector(o.inference.weights, o.inference.backend, o.inference.engine,
                                dataset.size());
    std::ofstream trajectory(o.output / "trajectory.tum"), frames(o.output / "frames.csv");
    if (!trajectory || !frames)
        throw std::runtime_error("Cannot write output files");
    trajectory
        << "# T_world_left_camera, dataset left-camera axes; timestamp tx ty tz qx qy qz qw\n";
    frames
        << "timestamp_ns,status,tracks,inliers,keyframe,io_ms,tracking_ms,pose_ms,keyframe_ms,total_ms,raw_tracks,exposure_retry,exposure_recovered,exposure_gain,exposure_bias,stereo_left,stereo_right,stereo_matches,stereo_geometry,stereo_refined,stereo_refinement_rejected,stereo_refinement_fallback,temporal_landmarks_before,temporal_lk_attempts,landmarks_after,landmark_cells,replenishment_reason,keyframe_attempted,stereo_landmarks,retained_landmarks,added_landmarks,selected_camera,left_raw_tracks,right_raw_tracks,left_tracks,right_tracks,left_inliers,right_inliers,left_pose_valid,right_pose_valid,left_pose_score,right_pose_score,left_joint_rmse,right_joint_rmse,left_exposure_retry,right_exposure_retry,left_exposure_recovered,right_exposure_recovered,left_exposure_gain,right_exposure_gain,left_exposure_bias,right_exposure_bias,joint_unique_inliers,pose_disagreement_m,pose_disagreement_deg,left_guided_attempted,right_guided_attempted,left_guided_tracks,right_guided_tracks,left_guided_recovered,right_guided_recovered,local_map_candidates,local_map_recovered,local_map_source_frame,local_map_size,stereo_solver_attempts,stereo_solver_valid,stereo_solver_unique_observations,stereo_solver_condition,stereo_solver_rejection,stereo_observation_set\n";
    std::ofstream stereoBaLog;
    struct AnchoredPose {
        TimestampNs timestamp;
        int anchor;
        cv::Matx44d anchorFromCamera;
    };
    std::vector<AnchoredPose> baHistory;
    std::map<int, cv::Matx44d> baAnchors;
    if (o.stereoBa.enabled) {
        stereoBaLog.open(o.output / "stereo_ba.csv");
        if (!stereoBaLog)
            throw std::runtime_error("Cannot write stereo BA diagnostics");
        stereoBaLog
            << "timestamp_ns,states,landmarks,observations,inliers,attempted,accepted,iterations,initial_cost,final_cost,rejection,ba_ms\n";
    }
    std::ofstream inertial;
    std::ofstream backend;
    std::ofstream residualAudit, imuWeightAudit;
    if (imu) {
        inertial.open(o.output / "inertial.csv");
        if (!inertial)
            throw std::runtime_error("Cannot write inertial diagnostics");
        inertial
            << "timestamp_ns,initialized,visual_update,reseed,position_std,innovation,vx,vy,vz,gx,gy,gz,bgx,bgy,bgz,bax,bay,baz,inertial_ms,recovery_phase,since_constraint_s,stationary_detected,zero_velocity_update,stationary_image_tracks,stationary_flow_median,stationary_gyro_rms,stationary_accel_std\n";
        if (o.vio.backend == VioBackend::Window) {
            backend.open(o.output / "backend.csv");
            if (!backend)
                throw std::runtime_error("Cannot write backend diagnostics");
            backend
                << "timestamp_ns,states,landmarks,observations,inliers,accepted,iterations,initial_cost,final_cost,prior_rank,marginalized_states,marginalized_landmarks,informative_landmarks,rejection,stationary_attempted,stationary_posterior_speed,stationary_nis,imu_gyro_noise_density_max,imu_accel_noise_density_max,solver_message,posterior_outliers,removed_observations,cleanup_attempted,cleanup_iterations,cleanup_initial_cost,cleanup_final_cost\n";
            if (o.vio.window.auditEvery > 0) {
                residualAudit.open(o.output / "window_residuals.csv");
                imuWeightAudit.open(o.output / "window_imu_weights.csv");
                if (!residualAudit || !imuWeightAudit)
                    throw std::runtime_error("Cannot write window audit");
                residualAudit
                    << std::setprecision(17)
                    << "timestamp_ns,frame,visual_accepted,group,blocks,dimensions,invalid_blocks,squared_norm,robust_squared_norm,max_abs\n";
                imuWeightAudit
                    << std::setprecision(17)
                    << "timestamp_ns,frame,first,second,seconds,model_uncertainty,whitening_error";
                for (const char* sensor : {"gyro_density_", "accel_density_"})
                    for (int k = 0; k < 3; ++k)
                        imuWeightAudit << ',' << sensor << k;
                for (const char* matrix : {"cov_diag_", "whitener_diag_"})
                    for (int k = 0; k < 15; ++k)
                        imuWeightAudit << ',' << matrix << k;
                imuWeightAudit << '\n';
            }
        }
    }
    const int requestedLimit = o.maxFrames > 0
                                   ? std::min(size_t(o.maxFrames), dataset.samples().size())
                                   : dataset.samples().size();
    int first = 0, limit = requestedLimit;
    if (imu) {
        while (first < limit && dataset.samples()[first].timestampNs < imu->firstTimestamp())
            ++first;
        while (limit > first && dataset.samples()[limit - 1].timestampNs > imu->lastTimestamp())
            --limit;
        if (first == limit)
            throw std::runtime_error("No image frames inside IMU coverage");
        std::cout << "IMU overlap: skipped leading=" << first
                  << " trailing=" << requestedLimit - limit << '\n';
    }
    cv::FileStorage meta((o.output / "run.yaml").string(), cv::FileStorage::WRITE);
    meta << "dataset" << dataset.root().string() << "format" << dataset.format() << "backend"
         << o.inference.backend << "weights" << o.inference.weights << "engine"
         << o.inference.engine << "opencv" << CV_VERSION << "threads" << o.threads << "image_scale"
         << o.imageScale << "source_width" << dataset.sourceSize().width << "source_height"
         << dataset.sourceSize().height << "image_width" << dataset.size().width << "image_height"
         << dataset.size().height << "left_camera_index" << dataset.leftCamera() << "scale_x"
         << double(dataset.size().width) / dataset.sourceSize().width << "scale_y"
         << double(dataset.size().height) / dataset.sourceSize().height << "network_width"
         << (dataset.size().width + 7) / 8 * 8 << "network_height"
         << (dataset.size().height + 7) / 8 * 8 << "max_points" << o.tracking.maxPoints
         << "min_inliers" << o.tracking.minInliers << "keyframe_interval"
         << o.tracking.keyframeInterval << "reserve_minimum" << 3 * o.tracking.minInliers
         << "reserve_max_depth_age" << 2 * o.tracking.keyframeInterval
         << "reserve_projection_threshold" << 1.5 << "reserve_retry_interval" << 2 << "threshold"
         << o.tracking.threshold << "requested_max_frames" << o.maxFrames << "temporal_method"
         << "stereo_lk" << "exposure_compensation" << 1 << "guided_recovery"
         << "local_normalization_motion_projection" << "local_normalization_sigma" << 8
         << "local_map_keyframes" << 5 << "local_map_max_age" << 2 * o.tracking.keyframeInterval
         << "local_map_min_coverage" << .25 << "pose_selection"
         << "unique_landmark_truncated_reprojection_coverage" << "pose_threshold_px" << 3
         << "pose_right_score_margin" << stereoPoseScoreMargin << "joint_pose_recovery" << 1
         << "joint_min_coverage" << .25 << "joint_min_condition_ratio" << 1e-6 << "stereo_method"
         << "superpoint" << "stereo_descriptor_distance" << .7 << "stereo_refinement" << 1
         << "keep_unrefined_stereo" << 1 << "unpaired_left" << int(dataset.unpairedLeft())
         << "unpaired_right" << int(dataset.unpairedRight()) << "K_rectified" << dataset.K()
         << "T_right_left_rectified" << cv::Mat(rightFromLeft) << "R_rectified_from_left"
         << cv::Mat(dataset.rectification());
    meta << "visual_ba" << (o.stereoBa.enabled ? "local_stereo" : "none");
    if (o.stereoBa.enabled)
        meta << "ba_keyframes" << o.stereoBa.keyframes << "ba_landmarks" << o.stereoBa.landmarks
             << "ba_iterations" << o.stereoBa.iterations << "ba_observation_noise" << "uniform_1px"
             << "ba_solver_budget_s" << .03 << "ba_gauge" << "oldest_pose_fixed_metric_stereo"
             << "ba_schedule" << "accepted_keyframes_synchronous"
             << "ba_history_policy" << "bounded_keyframes_no_marginalization_prior"
             << "ba_online_trajectory" << "trajectory.tum" << "ba_refined_trajectory"
             << "trajectory_ba_final.tum";
    if (dataset.format() == "tumvi")
        meta << "source_pixel_type" << "uint16_linear"
             << "intensity_conversion" << "opencv_imread_grayscale_uint16_shift8"
             << "rectification_model" << "opencv_fisheye_balance0_fov_scale1"
             << "camera_calibration_source"
             << (dataset.root().parent_path() / "dso/camchain.yaml").string()
             << "imu_calibration_source"
             << (dataset.root().parent_path() / "dso/imu_config.yaml").string();
    if (dataset.leftCamera() == 0)
        meta << "R_rectified_from_cam0" << cv::Mat(dataset.rectification());
    if (dataset.format() == "kitti")
        meta << "T_cam0_left" << cv::Mat(dataset.cam0FromLeft());
    if (calibration.T_body_raw)
        meta << "T_body_cam0" << cv::Mat(*calibration.T_body_raw);
    if (imu)
        meta << "fusion"
             << (o.vio.backend == VioBackend::Window ? "stereo_inertial_window"
                                                     : "relative_pose_clone_eskf")
             << "observation_noise"
             << (o.vio.noise == ObservationNoise::Patch ? "bounded_patch_hessian" : "uniform_1px")
             << "window_states" << o.vio.window.states << "window_landmarks"
             << o.vio.window.landmarks << "window_feedback"
             << (o.vio.window.resolveAfterPruning   ? "prune_and_resolve"
                 : o.vio.window.observationFeedback ? "persistent_outlier_pruning"
                                                    : "off")
             << "window_feedback_outlier_chi2" << 9.21 << "window_feedback_pixel_cap" << 3.
             << "window_feedback_streak" << 2 << "window_feedback_max_removed" << 32
             << "window_feedback_removed_fraction" << .1 << "window_cleanup_iterations"
             << (o.vio.window.resolveAfterPruning ? 2 : 0) << "window_iterations"
             << o.vio.window.iterations << "imu_rate_hz" << imu->calibration().rateHz
             << "window_audit_every" << o.vio.window.auditEvery << "window_gravity_model"
             << "joint_vector_with_magnitude_prior" << "window_initial_gravity_std" << .2
             << "window_gravity_magnitude_std" << .01 << "T_imu_rect"
             << cv::Mat(imu->calibration().T_imu_rect) << "max_inertial_only_seconds"
             << RecoveryPolicy::maxPropagationSeconds << "max_relative_position_std"
             << RecoveryPolicy::maxRelativePositionStd << "max_relative_rotation_std"
             << RecoveryPolicy::maxRelativeRotationStd << "zero_velocity_update"
             << "image_imu_consensus" << "stationary_window_seconds" << .3
             << "imu_skipped_leading_frames" << first << "imu_skipped_trailing_frames"
             << requestedLimit - limit << "gyro_noise_density" << imu->calibration().gyroNoise
             << "accel_noise_density" << imu->calibration().accelNoise << "gyro_random_walk"
             << imu->calibration().gyroWalk << "accel_random_walk" << imu->calibration().accelWalk;
    if (o.loop.enabled)
        meta << "loop_closure" << "superpoint_mean_stereo_4dof" << "loop_synchronous"
             << int(o.loop.synchronous) << "loop_keyframe_seconds" << o.loop.keyframeSeconds
             << "loop_exclusion_seconds" << o.loop.exclusionSeconds << "loop_candidates"
             << o.loop.candidates << "loop_min_inliers" << o.loop.minInliers
             << "loop_pixel_threshold" << o.loop.pixelThreshold;
    meta.release();
    auto tracking = o.tracking;
    tracking.retainFeatures = o.loop.enabled;
    Odometry estimator(calibration, tracking, detector,
                       imu ? std::optional<ImuCalibration>(imu->calibration()) : std::nullopt,
                       o.vio, o.stereoBa);
    int exitCode = 0;
#ifdef SUPERVO_WITH_LOOP
    std::unique_ptr<LoopClosure> loop;
    std::ofstream mapOnline, corrections;
    std::vector<std::pair<TimestampNs, cv::Matx44d>> localPoses;
    if (o.loop.enabled) {
        loop = std::make_unique<LoopClosure>(calibration, o.loop);
        mapOnline.open(o.output / "trajectory_map_online.tum");
        corrections.open(o.output / "loop_versions.csv");
        corrections << "timestamp_ns,version,anchor_ns\n";
        if (!mapOnline || !corrections)
            throw std::runtime_error("Cannot open loop outputs");
    }
#endif
    auto wallStart = Clock::now();
    for (int i = first; i < limit; ++i) {
        auto start = Clock::now();
        auto images = dataset.read(i);
        double ioTime = ms(start);
        const auto& sample = dataset.samples()[i];
        const auto interval = imu ? imu->interval(i > first ? dataset.samples()[i - 1].timestampNs
                                                            : sample.timestampNs,
                                                  sample.timestampNs)
                                  : std::vector<ImuSample>{};
        const auto result = estimator.process({sample.timestampNs, images}, interval);
        const auto& d = result.diagnostics;
        double total = ms(start);
        frames << sample.timestampNs << ','
               << (result.status == TrackingStatus::InitializingInertial  ? "INITIALIZING"
                   : result.status == TrackingStatus::StationaryInertial  ? "STATIONARY"
                   : result.status == TrackingStatus::PropagatingInertial ? "PROPAGATING"
                   : d.valid                                              ? "TRACKING"
                   : i == first                                           ? "INIT_FAILED"
                                                                          : "LOST")
               << ',' << d.count << ',' << d.inliers << ',' << d.key << ',' << ioTime << ','
               << d.trackTime << ',' << d.poseTime << ',' << d.keyTime << ',' << total << ','
               << d.rawCount << ',' << d.exposureRetry << ',' << d.exposureRecovered << ','
               << d.exposureGain << ',' << d.exposureBias << ',' << d.stereoStats.left << ','
               << d.stereoStats.right << ',' << d.stereoStats.matches << ','
               << d.stereoStats.geometry << ',' << d.stereoStats.refined << ','
               << d.stereoStats.refinementRejected << ',' << d.stereoStats.refinementFallback << ','
               << d.landmarksBefore << ',' << d.lkAttempts << ',' << d.landmarksAfter << ','
               << d.cells << ',' << d.reason << ',' << d.attempted << ',' << d.stereoLandmarks
               << ',' << d.retained << ',' << d.added << ','
               << (i == first                         ? "init"
                   : d.selected == PoseSource::Left   ? "left"
                   : d.selected == PoseSource::Right  ? "right"
                   : d.selected == PoseSource::Stereo ? "stereo"
                                                      : "none")
               << ',' << d.rawCounts[0] << ',' << d.rawCounts[1] << ',' << d.tracks[0].pixels.size()
               << ',' << d.tracks[1].pixels.size() << ',' << d.candidates[0].inliers.size() << ','
               << d.candidates[1].inliers.size() << ',' << d.candidates[0].valid << ','
               << d.candidates[1].valid << ',' << d.quality[0].score << ',' << d.quality[1].score
               << ',' << d.quality[0].rmse << ',' << d.quality[1].rmse << ',' << d.retries[0] << ','
               << d.retries[1] << ',' << d.recoveries[0] << ',' << d.recoveries[1] << ','
               << d.gains[0] << ',' << d.gains[1] << ',' << d.biases[0] << ',' << d.biases[1] << ','
               << d.jointInliers << ',' << d.disagreementM << ',' << d.disagreementDeg << ','
               << d.guidedAttempted[0] << ',' << d.guidedAttempted[1] << ',' << d.guidedCounts[0]
               << ',' << d.guidedCounts[1] << ',' << d.guidedRecovered[0] << ','
               << d.guidedRecovered[1] << ',' << d.mapCandidates << ',' << d.mapRecovered << ','
               << d.mapSource << ',' << d.mapSize << ',' << d.stereoAttempts << ','
               << d.jointEstimate.pose.valid << ',' << d.jointEstimate.uniqueObservations << ','
               << d.jointEstimate.conditionRatio << ',' << d.jointEstimate.rejection << ','
               << d.stereoObservationSet << '\n';
        if (result.inertial) {
            const auto& state = *result.inertial;
            inertial << sample.timestampNs << ',' << state.initialized << ','
                     << result.inertialUpdate << ',' << result.inertialReseed << ','
                     << state.positionStd << ',' << state.innovation;
            for (const auto& value :
                 {state.velocity, state.gravity, state.gyroBias, state.accelBias})
                for (double x : value.val)
                    inertial << ',' << x;
            inertial << ',' << result.inertialMs << ',' << static_cast<int>(result.recoveryPhase)
                     << ',' << result.sinceConstraintSeconds << ',' << result.stationary.stationary
                     << ',' << result.zeroVelocityUpdate << ',' << result.stationary.imageTracks
                     << ',' << result.stationary.flowMedian << ',' << result.stationary.gyroRms
                     << ',' << result.stationary.accelerationStd << '\n';
        }
        if (backend.is_open()) {
            const auto& b = result.backend;
            backend << sample.timestampNs << ',' << b.states << ',' << b.landmarks << ','
                    << b.observations << ',' << b.inliers << ',' << b.accepted << ','
                    << b.iterations << ',' << b.initialCost << ',' << b.finalCost << ','
                    << b.priorRank << ',' << b.marginalizedStates << ',' << b.marginalizedLandmarks
                    << ',' << b.informativeLandmarks << ',' << b.rejection << ','
                    << b.stationaryAttempted << ',' << b.stationaryPosteriorSpeed << ','
                    << b.stationaryInnovation << ',' << b.imuGyroNoise << ',' << b.imuAccelNoise
                    << ",\"";
            for (char c : b.solverMessage) {
                if (c == '\"')
                    backend << '\"';
                backend << c;
            }
            backend << "\"," << b.posteriorOutliers << ',' << b.removedObservations << ','
                    << b.cleanupAttempted << ',' << b.cleanupIterations << ','
                    << b.cleanupInitialCost << ',' << b.cleanupFinalCost << '\n';
            if (b.audit) {
                const auto& a = *b.audit;
                for (size_t k = 0; k < a.residuals.size(); ++k) {
                    const auto& r = a.residuals[k];
                    residualAudit << sample.timestampNs << ',' << a.frame << ',' << b.accepted
                                  << ',' << windowAuditGroups[k] << ',' << r.blocks << ','
                                  << r.dimensions << ',' << r.invalidBlocks << ',' << r.squaredNorm
                                  << ',' << r.robustSquaredNorm << ',' << r.maxAbs << '\n';
                }
                for (const auto& w : a.imu) {
                    imuWeightAudit << sample.timestampNs << ',' << a.frame << ',' << w.first << ','
                                   << w.second << ',' << w.seconds << ',' << w.modelUncertainty
                                   << ',' << w.whiteningError;
                    for (double value : w.gyroDensity)
                        imuWeightAudit << ',' << value;
                    for (double value : w.accelDensity)
                        imuWeightAudit << ',' << value;
                    for (double value : w.covarianceDiagonal)
                        imuWeightAudit << ',' << value;
                    for (double value : w.whitenerDiagonal)
                        imuWeightAudit << ',' << value;
                    imuWeightAudit << '\n';
                }
            }
        }
        if (stereoBaLog.is_open()) {
            const auto& b = result.stereoBa;
            stereoBaLog << sample.timestampNs << ',' << b.states << ',' << b.landmarks << ','
                        << b.observations << ',' << b.inliers << ',' << b.attempted << ','
                        << b.accepted << ',' << b.iterations << ',' << b.initialCost << ','
                        << b.finalCost << ',' << b.rejection << ',' << result.stereoBaMs << '\n';
        }
        if (!result.T_world_rect) {
            frames.flush();
            trajectory.flush();
            std::cerr << "Tracking failed at frame " << i
                      << "; partial trajectory retained, no reset or invented pose\n";
            exitCode = 2;
            break;
        }
        writePose(trajectory, sample.timestampNs, calibration.rawPose(*result.T_world_rect));
        if (o.stereoBa.enabled) {
            for (const auto& update : result.baPoses)
                baAnchors[update.frame] = update.worldFromCamera;
            const auto anchor = baAnchors.at(result.baReferenceFrame);
            baHistory.push_back({sample.timestampNs, result.baReferenceFrame,
                                 anchor.inv(cv::DECOMP_SVD) * *result.T_world_rect});
        }
#ifdef SUPERVO_WITH_LOOP
        if (loop) {
            if (result.keyframeFeatures && result.inertial && result.inertial->initialized)
                loop->submit({result.frameIndex, sample.timestampNs, *result.T_world_rect,
                              result.inertial->gravity, result.keyframeFeatures});
            auto correction = loop->correction();
            writePose(mapOnline, sample.timestampNs,
                      calibration.rawPose(correction.mapFromOdom * *result.T_world_rect));
            corrections << sample.timestampNs << ',' << correction.version << ','
                        << correction.anchor << '\n';
            localPoses.emplace_back(sample.timestampNs, *result.T_world_rect);
        }
#endif

        if (i % 100 == 0 || i == limit - 1)
            std::cout << "frame=" << i << " inliers=" << d.inliers
                      << " landmarks=" << d.landmarksAfter << " keyframe=" << d.key << " camera="
                      << (d.selected == PoseSource::Stereo  ? "stereo"
                          : d.selected == PoseSource::Right ? "right"
                                                            : "left")
                      << " total_ms=" << total << std::endl;
        if (o.realtime) {
            auto target =
                wallStart +
                std::chrono::nanoseconds(sample.timestampNs - dataset.samples()[first].timestampNs);
            std::this_thread::sleep_until(target);
        }
    }
#ifdef SUPERVO_WITH_LOOP
    if (loop) {
        auto report = loop->finish();
        std::ofstream finalTrajectory(o.output / "trajectory_map_final.tum"),
            candidates(o.output / "loop_candidates.csv"), edges(o.output / "loop_edges.csv");
        candidates
            << "current,reference,similarity,matches,inliers,rmse,coverage,tilt,rejection,decision,query_elapsed_ms\n";
        for (const auto& c : report.trials)
            candidates << c.current << ',' << c.reference << ',' << c.similarity << ','
                       << c.geometry.matches << ',' << c.geometry.inliers << ',' << c.geometry.rmse
                       << ',' << c.geometry.coverage << ',' << c.geometry.tilt << ','
                       << c.geometry.rejection << ',' << c.decision << ',' << c.milliseconds
                       << '\n';
        edges << "first,second,loop,switch_weight\n";
        for (const auto& e : report.edges)
            edges << e.first << ',' << e.second << ',' << e.loop << ',' << e.weight << '\n';
        size_t anchor = 0;
        for (const auto& [timestamp, pose] : localPoses) {
            cv::Matx44d correction = cv::Matx44d::eye();
            if (!report.keyframes.empty() && timestamp >= report.keyframes.front().timestamp) {
                while (anchor + 1 < report.keyframes.size() &&
                       report.keyframes[anchor + 1].timestamp <= timestamp)
                    ++anchor;
                correction = report.globalPoses[anchor] *
                             report.keyframes[anchor].odomPose.inv(cv::DECOMP_SVD);
            }
            writePose(finalTrajectory, timestamp, calibration.rawPose(correction * pose));
        }
        std::filesystem::create_directories(o.output / "loop_keyframes");
        for (size_t k = 0; k < report.keyframes.size(); ++k) {
            const auto& key = report.keyframes[k];
            const auto& f = *key.features;
            cv::FileStorage archive(
                (o.output / "loop_keyframes" / (std::to_string(key.frame) + ".yml.gz")).string(),
                cv::FileStorage::WRITE);
            archive << "frame" << key.frame << "timestamp_ns" << std::to_string(key.timestamp)
                    << "odom_pose" << cv::Mat(key.odomPose) << "map_pose"
                    << cv::Mat(report.globalPoses[k]) << "pixels" << f.pixels << "right_pixels"
                    << f.rightPixels << "points" << f.points << "depth_valid" << f.depthValid
                    << "descriptors" << f.descriptors;
            if (!archive.isOpened())
                throw std::runtime_error("Cannot save feature archive");
        }
        cv::FileStorage summary((o.output / "loop_summary.yaml").string(), cv::FileStorage::WRITE);
        summary << "keyframes" << int(report.keyframes.size()) << "graph_versions"
                << int(report.version) << "queue_peak" << int(report.queuePeak) << "aggregation"
                << "mean_l2" << "global_dof" << 4;
        mapOnline.flush();
        corrections.flush();
        finalTrajectory.flush();
        candidates.flush();
        edges.flush();
        if (!mapOnline || !corrections || !finalTrajectory || !candidates || !edges)
            throw std::runtime_error("Loop output write failed");
    }
#endif
    if (o.stereoBa.enabled) {
        std::ofstream refined(o.output / "trajectory_ba_final.tum");
        if (!refined)
            throw std::runtime_error("Cannot write refined stereo BA trajectory");
        refined << "# Historical keyframe corrections; not the online published trajectory\n";
        for (const auto& record : baHistory)
            writePose(refined, record.timestamp,
                      calibration.rawPose(baAnchors.at(record.anchor) * record.anchorFromCamera));
        refined.flush();
        stereoBaLog.flush();
        if (!refined || !stereoBaLog)
            throw std::runtime_error("Writing stereo BA results failed");
    }
    trajectory.flush();
    frames.flush();
    if (imu) {
        inertial.flush();
        if (!inertial)
            throw std::runtime_error("Writing inertial diagnostics failed");
    }
    if (backend.is_open()) {
        backend.flush();
        if (!backend)
            throw std::runtime_error("Writing backend diagnostics failed");
    }
    if (!trajectory || !frames)
        throw std::runtime_error("Writing trajectory failed");
    return exitCode;
}
}
