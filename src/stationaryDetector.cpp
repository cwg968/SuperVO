#include <stationaryDetector.h>
#include <frame.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <set>
namespace superVO {
StationaryEvidence StationaryDetector::observe(const StereoFrame& frame,
                                               const std::vector<ImuSample>& imu,
                                               const InertialState& state,
                                               const ImuCalibration& calibration) {
    StationaryEvidence result;
    const double dt = (frame.timestampNs - timestamp_) * 1e-9;
    bool imageStill = false;
    if (timestamp_ >= 0 && dt > 0 && dt <= .15) {
        for (int camera = 0; camera < 2; ++camera) {
            const auto& before = camera ? previous_.right : previous_.left;
            const auto& now = camera ? frame.images.right : frame.images.left;
            std::vector<cv::Point2f> points;
            cv::goodFeaturesToTrack(before, points, 160, .01, 10);
            const auto tracks = trackGuidedPoints(before, now, points, points, false);
            std::vector<double> flow;
            std::set<int> cells;
            for (size_t j = 0; j < tracks.pixels.size(); ++j) {
                const auto origin = points.at(tracks.sourceIndices[j]);
                flow.push_back(cv::norm(tracks.pixels[j] - origin) * .05 / dt);
                cells.insert(std::min(3, int(origin.y * 4 / before.rows)) * 6 +
                             std::min(5, int(origin.x * 6 / before.cols)));
            }
            if (flow.size() < 16 || cells.size() < 4)
                continue;
            std::sort(flow.begin(), flow.end());
            if (int(flow.size()) > result.imageTracks) {
                result.imageTracks = flow.size();
                result.flowMedian = flow[flow.size() / 2];
            }
            imageStill |= flow[flow.size() / 2] < .35 && flow[flow.size() * 9 / 10] < .8;
        }
    }
    previous_ = {frame.images.left.clone(), frame.images.right.clone()};
    timestamp_ = frame.timestampNs;
    motion_.push_back({timestamp_, imageStill});
    for (const auto& sample : imu)
        if (samples_.empty() || sample.timestampNs > samples_.back().timestampNs)
            samples_.push_back(sample);
    const auto cutoff = timestamp_ - 300000000;
    while (!samples_.empty() && samples_.front().timestampNs < cutoff)
        samples_.pop_front();
    while (!motion_.empty() && motion_.front().first < cutoff)
        motion_.pop_front();
    if (!state.initialized || samples_.size() < 2 ||
        timestamp_ - samples_.front().timestampNs < 250000000)
        return result;
    double weight = 0, gyroSquared = 0;
    cv::Vec3d accelSquared{};
    for (size_t j = 1; j < samples_.size(); ++j) {
        const auto& a = samples_[j - 1];
        const auto& b = samples_[j];
        const double duration = (b.timestampNs - a.timestampNs) * 1e-9;
        const auto gyro = .5 * (a.angularVelocity + b.angularVelocity);
        const auto accel = .5 * (a.acceleration + b.acceleration);
        result.meanAngularVelocity += gyro * duration;
        result.meanAcceleration += accel * duration;
        gyroSquared += (gyro - state.gyroBias).dot(gyro - state.gyroBias) * duration;
        for (int k = 0; k < 3; ++k)
            accelSquared[k] += accel[k] * accel[k] * duration;
        weight += duration;
    }
    result.meanAngularVelocity /= weight;
    result.meanAcceleration /= weight;
    result.gyroRms = std::sqrt(gyroSquared / weight);
    double variance = 0;
    for (int k = 0; k < 3; ++k)
        variance += std::max(0., accelSquared[k] / weight -
                                     result.meanAcceleration[k] * result.meanAcceleration[k]);
    result.accelerationStd = std::sqrt(variance);
    const auto body = state.T_world_rect * calibration.T_imu_rect.inv(cv::DECOMP_SVD);
    cv::Matx33d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = body(r, c);
    const auto specificResidual = R * (result.meanAcceleration - state.accelBias) + state.gravity;
    result.stationary =
        !motion_.empty() && timestamp_ - motion_.front().first >= 200000000 &&
        std::all_of(motion_.begin(), motion_.end(), [](const auto& m) { return m.second; }) &&
        result.gyroRms < .04 && result.accelerationStd < .55 && cv::norm(specificResidual) < .6 &&
        cv::norm(state.velocity) < .25;
    return result;
}
}
