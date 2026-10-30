#include <inertialEstimator.h>
#include "inertialMath.h"
#include <deque>
#include <stdexcept>
namespace superVO {
using namespace inertial;
struct InertialEstimator::Impl {
    ImuCalibration calibration;
    bool ready = false;
    TimestampNs timestamp = 0;
    M3 R = M3::Identity();
    V3 p = V3::Zero(), v = V3::Zero(), bg = V3::Zero(), ba = V3::Zero(), g = V3::Zero();
    MX P;
    double innovation = 0;
    int protectedFrame = -1;
    struct Clone {
        int frame;
        M4 camera;
    };
    std::deque<Clone> clones;
    explicit Impl(ImuCalibration c) : calibration(c) {
        c.validate();
    }
    M4 camera() const {
        return pose(R, p) * eigen(calibration.T_imu_rect);
    }
    M4 relative(size_t index) const {
        return inverse(clones[index].camera) * camera();
    }
    void inject(const VX& dx) {
        p += dx.segment<3>(0);
        v += dx.segment<3>(3);
        R = R * exp(dx.segment<3>(6));
        bg += dx.segment<3>(9);
        ba += dx.segment<3>(12);
        g += dx.segment<3>(15);
        for (size_t j = 0; j < clones.size(); ++j) {
            auto& T = clones[j].camera;
            const int offset = 18 + 6 * j;
            T.topRightCorner<3, 1>() += dx.segment<3>(offset);
            T.topLeftCorner<3, 3>() = T.topLeftCorner<3, 3>() * exp(dx.segment<3>(offset + 3));
        }
    }
    bool update(const MX& H, const VX& residual, const MX& noise, double gate, double& nis) {
        const int n = P.rows();
        const MX S = H * P * H.transpose() + noise;
        Eigen::LDLT<MX> factor(S);
        if (factor.info() != Eigen::Success || !factor.isPositive())
            return false;
        nis = residual.dot(factor.solve(residual));
        if (!std::isfinite(nis) || nis > gate)
            return false;
        const MX gain = factor.solve(H * P).transpose();
        const VX dx = gain * residual;
        if (!dx.allFinite())
            return false;
        if ((g + dx.segment<3>(15)).norm() < 1.)
            return false;
        const MX I = MX::Identity(n, n), A = I - gain * H;
        P = A * P * A.transpose() + gain * noise * gain.transpose();
        inject(dx);
        MX reset = I;
        reset.block<3, 3>(6, 6) -= .5 * skew(dx.segment<3>(6));
        for (size_t j = 0; j < clones.size(); ++j)
            reset.block<3, 3>(21 + 6 * j, 21 + 6 * j) -= .5 * skew(dx.segment<3>(21 + 6 * j));
        // Gravity has fixed magnitude; project its covariance onto the tangent plane.
        const double magnitude = g.norm();
        const V3 direction = g / magnitude;
        reset.block<3, 3>(15, 15) =
            (9.81 / magnitude) * (M3::Identity() - direction * direction.transpose());
        g = 9.81 * direction;
        P = reset * P * reset.transpose();
        P = .5 * (P + P.transpose()).eval();
        return true;
    }
};
InertialEstimator::InertialEstimator(ImuCalibration c) : impl_(std::make_unique<Impl>(c)) {}
InertialEstimator::~InertialEstimator() = default;
InertialEstimator::InertialEstimator(const InertialEstimator& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}
InertialEstimator& InertialEstimator::operator=(const InertialEstimator& other) {
    if (this != &other)
        impl_ = std::make_unique<Impl>(*other.impl_);
    return *this;
}
bool InertialEstimator::initialized() const {
    return impl_->ready;
}
void InertialEstimator::initialize(const InertialState& initial, TimestampNs timestamp) {
    auto& s = *impl_;
    if (s.ready || timestamp < 0 || !initial.initialized ||
        !cv::checkRange(cv::Mat(initial.T_world_rect)))
        throw std::invalid_argument("Invalid inertial initialization");
    const M4 body = eigen(initial.T_world_rect) * inverse(eigen(s.calibration.T_imu_rect));
    s.R = body.topLeftCorner<3, 3>();
    s.p = body.topRightCorner<3, 1>();
    s.v = eigen(initial.velocity);
    s.g = eigen(initial.gravity);
    s.bg = eigen(initial.gyroBias);
    s.ba = eigen(initial.accelBias);
    if (!s.v.allFinite() || !s.g.allFinite() || !s.bg.allFinite() || !s.ba.allFinite() ||
        std::abs(s.g.norm() - 9.81) > .1)
        throw std::invalid_argument("Invalid initial gravity or velocity");
    s.P = MX::Zero(18, 18);
    const double sigmas[] = {.001, .2, .01, .02, .1, .2};
    for (int j = 0; j < 6; ++j)
        s.P.block<3, 3>(j * 3, j * 3) = M3::Identity() * sigmas[j] * sigmas[j];
    s.timestamp = timestamp;
    s.ready = true;
}
void InertialEstimator::propagate(const std::vector<ImuSample>& imu) {
    auto& s = *impl_;
    if (!s.ready || imu.size() < 2 || imu.front().timestampNs != s.timestamp)
        throw std::invalid_argument("IMU propagation interval is not contiguous");
    validateImuInterval(imu, s.timestamp, imu.back().timestampNs, s.calibration.rateHz);
    for (size_t j = 1; j < imu.size(); ++j) {
        const double dt = (imu[j].timestampNs - imu[j - 1].timestampNs) * 1e-9;
        const V3 omega =
            .5 * (eigen(imu[j].angularVelocity) + eigen(imu[j - 1].angularVelocity)) - s.bg;
        const V3 specific =
            .5 * (eigen(imu[j].acceleration) + eigen(imu[j - 1].acceleration)) - s.ba;
        const M3 mid = s.R * exp(omega * (.5 * dt));
        const V3 acceleration = mid * specific + s.g;
        MX F = MX::Zero(18, 18);
        F.block<3, 3>(0, 3) = M3::Identity();
        F.block<3, 3>(3, 6) = -mid * skew(specific);
        F.block<3, 3>(3, 12) = -mid;
        F.block<3, 3>(3, 15) = M3::Identity();
        F.block<3, 3>(6, 6) = -skew(omega);
        F.block<3, 3>(6, 9) = -M3::Identity();
        const MX phi = MX::Identity(18, 18) + F * dt + .5 * F * F * dt * dt;
        MX Q = MX::Zero(18, 18);
        const auto& c = s.calibration;
        Q.block<3, 3>(3, 3) = M3::Identity() * c.accelNoise * c.accelNoise;
        Q.block<3, 3>(6, 6) = M3::Identity() * c.gyroNoise * c.gyroNoise;
        Q.block<3, 3>(9, 9) = M3::Identity() * c.gyroWalk * c.gyroWalk;
        Q.block<3, 3>(12, 12) = M3::Identity() * c.accelWalk * c.accelWalk;
        const MX half = MX::Identity(18, 18) + .5 * F * dt;
        const MX propagated =
            phi * s.P.topLeftCorner(18, 18) * phi.transpose() + half * Q * half.transpose() * dt;
        if (s.P.cols() > 18) {
            const MX cross = phi * s.P.topRightCorner(18, s.P.cols() - 18);
            s.P.topRightCorner(18, s.P.cols() - 18) = cross;
            s.P.bottomLeftCorner(s.P.rows() - 18, 18) = cross.transpose();
        }
        s.P.topLeftCorner(18, 18) = propagated;
        s.p += s.v * dt + .5 * acceleration * dt * dt;
        s.v += acceleration * dt;
        s.R = s.R * exp(omega * dt);
    }
    s.P = .5 * (s.P + s.P.transpose()).eval();
    s.timestamp = imu.back().timestampNs;
}
bool InertialEstimator::correct(const cv::Matx44d& currentFromReference, int referenceFrame,
                                double rmse, size_t inliers) {
    auto& s = *impl_;
    size_t index = 0;
    while (index < s.clones.size() && s.clones[index].frame != referenceFrame)
        ++index;
    if (!s.ready || index == s.clones.size() || inliers < 6 || !std::isfinite(rmse))
        return false;
    const M4 prediction = s.relative(index), measurement = inverse(eigen(currentFromReference));
    VX residual(6);
    residual.head<3>() = measurement.topRightCorner<3, 1>() - prediction.topRightCorner<3, 1>();
    residual.tail<3>() =
        log(prediction.topLeftCorner<3, 3>().transpose() * measurement.topLeftCorner<3, 3>());
    const int n = s.P.rows();
    MX H = MX::Zero(6, n);
    const double epsilon = 1e-6;
    // Differentiate only the current body pose and the observed camera clone.
    for (int j : {0, 1, 2, 6, 7, 8, int(18 + index * 6), int(19 + index * 6), int(20 + index * 6),
                  int(21 + index * 6), int(22 + index * 6), int(23 + index * 6)}) {
        auto perturbed = s;
        VX dx = VX::Zero(n);
        dx[j] = epsilon;
        perturbed.inject(dx);
        const M4 value = perturbed.relative(index);
        H.block<3, 1>(0, j) =
            (value.topRightCorner<3, 1>() - prediction.topRightCorner<3, 1>()) / epsilon;
        H.block<3, 1>(3, j) =
            log(prediction.topLeftCorner<3, 3>().transpose() * value.topLeftCorner<3, 3>()) /
            epsilon;
    }
    // Pose-level fusion uses a conservative fixed noise floor; not a claim of calibrated PnP covariance.
    const double scale = std::max(1., rmse) * std::sqrt(std::max(1., 30. / double(inliers)));
    MX noise = MX::Zero(6, 6);
    noise.topLeftCorner<3, 3>() = M3::Identity() * std::pow(.015 * scale, 2);
    noise.bottomRightCorner<3, 3>() = M3::Identity() * std::pow(.004 * scale, 2);
    return s.update(H, residual, noise, 50, s.innovation);
}
bool InertialEstimator::correctStationary(const cv::Vec3d& meanGyro, const cv::Vec3d& meanAccel) {
    auto& s = *impl_;
    if (!s.ready || !cv::checkRange(cv::Mat(meanGyro)) || !cv::checkRange(cv::Mat(meanAccel)))
        return false;
    MX H = MX::Zero(9, s.P.cols());
    VX residual(9);
    H.block<3, 3>(0, 3) = M3::Identity();
    residual.segment<3>(0) = -s.v;
    H.block<3, 3>(3, 9) = M3::Identity();
    residual.segment<3>(3) = eigen(meanGyro) - s.bg;
    const V3 gravityBody = s.R.transpose() * s.g;
    H.block<3, 3>(6, 6) = -skew(gravityBody);
    H.block<3, 3>(6, 12) = M3::Identity();
    H.block<3, 3>(6, 15) = -s.R.transpose();
    residual.segment<3>(6) = eigen(meanAccel) + gravityBody - s.ba;
    MX noise = MX::Zero(9, 9);
    noise.block<3, 3>(0, 0) = .03 * .03 * M3::Identity();
    noise.block<3, 3>(3, 3) = .008 * .008 * M3::Identity();
    noise.block<3, 3>(6, 6) = .3 * .3 * M3::Identity();
    double nis = 0;
    return s.update(H, residual, noise, 40, nis);
}

void InertialEstimator::rememberReference(int frameIndex) {
    auto& s = *impl_;
    if (!s.ready)
        throw std::logic_error("Cannot clone an uninitialized filter");
    for (const auto& c : s.clones)
        if (c.frame == frameIndex)
            throw std::logic_error("Duplicate inertial clone");
    const int n = s.P.rows();
    MX J = MX::Zero(6, n);
    J.block<3, 3>(0, 0) = M3::Identity();
    const M4 E = eigen(s.calibration.T_imu_rect);
    J.block<3, 3>(0, 6) = -s.R * skew(E.topRightCorner<3, 1>());
    J.block<3, 3>(3, 6) = E.topLeftCorner<3, 3>().transpose();
    MX augmented = MX::Zero(n + 6, n + 6);
    augmented.topLeftCorner(n, n) = s.P;
    augmented.bottomLeftCorner(6, n) = J * s.P;
    augmented.topRightCorner(n, 6) = s.P * J.transpose();
    augmented.bottomRightCorner(6, 6) = J * s.P * J.transpose();
    s.P = std::move(augmented);
    s.clones.push_back({frameIndex, s.camera()});
    if (s.clones.size() > 6) {
        const size_t removed = s.clones.front().frame == s.protectedFrame ? 1 : 0;
        const int offset = 18 + 6 * removed, m = s.P.rows();
        std::vector<int> keep;
        for (int j = 0; j < m; ++j)
            if (j < offset || j >= offset + 6)
                keep.push_back(j);
        MX reduced(m - 6, m - 6);
        for (int r = 0; r < m - 6; ++r)
            for (int c = 0; c < m - 6; ++c)
                reduced(r, c) = s.P(keep[r], keep[c]);
        s.P = std::move(reduced);
        s.clones.erase(s.clones.begin() + removed);
    }
}
void InertialEstimator::protectReference(int frameIndex) {
    if (!referencePose(frameIndex))
        throw std::logic_error("Cannot protect missing reference");
    impl_->protectedFrame = frameIndex;
}
std::pair<double, double> InertialEstimator::relativeUncertainty(int frameIndex) const {
    const auto& s = *impl_;
    size_t index = 0;
    while (index < s.clones.size() && s.clones[index].frame != frameIndex)
        ++index;
    if (!s.ready || index == s.clones.size())
        return {INFINITY, INFINITY};
    const M4 prediction = s.relative(index);
    const int n = s.P.rows();
    MX H = MX::Zero(6, n);
    for (int j : {0, 1, 2, 6, 7, 8, int(18 + index * 6), int(19 + index * 6), int(20 + index * 6),
                  int(21 + index * 6), int(22 + index * 6), int(23 + index * 6)}) {
        auto perturbed = s;
        VX dx = VX::Zero(n);
        dx[j] = 1e-6;
        perturbed.inject(dx);
        const M4 value = perturbed.relative(index);
        H.block<3, 1>(0, j) =
            (value.topRightCorner<3, 1>() - prediction.topRightCorner<3, 1>()) / 1e-6;
        H.block<3, 1>(3, j) =
            log(prediction.topLeftCorner<3, 3>().transpose() * value.topLeftCorner<3, 3>()) / 1e-6;
    }
    const MX covariance = H * s.P * H.transpose();
    return {std::sqrt(std::max(0., covariance.topLeftCorner<3, 3>().trace())),
            std::sqrt(std::max(0., covariance.bottomRightCorner<3, 3>().trace()))};
}
std::optional<cv::Matx44d> InertialEstimator::referencePose(int frameIndex) const {
    for (const auto& c : impl_->clones)
        if (c.frame == frameIndex)
            return cvpose(c.camera);
    return std::nullopt;
}
InertialState InertialEstimator::state() const {
    const auto& s = *impl_;
    InertialState result;
    result.initialized = s.ready;
    if (!s.ready)
        return result;
    result.T_world_rect = cvpose(s.camera());
    result.velocity = cvvec(s.v);
    result.gravity = cvvec(s.g);
    result.gyroBias = cvvec(s.bg);
    result.accelBias = cvvec(s.ba);
    result.positionStd = std::sqrt(s.P.topLeftCorner<3, 3>().trace());
    result.innovation = s.innovation;
    return result;
}
}
