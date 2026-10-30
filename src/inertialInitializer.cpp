#include <inertialEstimator.h>
#include "inertialMath.h"
#include <deque>
#include <stdexcept>
namespace superVO {
using namespace inertial;
struct InertialInitializer::Impl {
    ImuCalibration calibration;
    struct Frame {
        M4 body;
        TimestampNs timestamp;
        std::vector<ImuSample> imu;
    };
    std::deque<Frame> frames;
    explicit Impl(ImuCalibration c) : calibration(c) {
        c.validate();
    }
};
InertialInitializer::InertialInitializer(ImuCalibration c) : impl_(std::make_unique<Impl>(c)) {}
InertialInitializer::~InertialInitializer() = default;
InertialInitializer::InertialInitializer(const InertialInitializer& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}
InertialInitializer& InertialInitializer::operator=(const InertialInitializer& other) {
    if (this != &other)
        impl_ = std::make_unique<Impl>(*other.impl_);
    return *this;
}
std::optional<InertialState> InertialInitializer::observe(const cv::Matx44d& camera,
                                                          TimestampNs timestamp,
                                                          const std::vector<ImuSample>& imu) {
    auto& frames = impl_->frames;
    validateImuInterval(imu, frames.empty() ? timestamp : frames.back().timestamp, timestamp,
                        impl_->calibration.rateHz);
    if (!frames.empty() &&
        (timestamp <= frames.back().timestamp || imu.size() < 2 ||
         imu.front().timestampNs != frames.back().timestamp || imu.back().timestampNs != timestamp))
        throw std::invalid_argument("Initializer requires contiguous image/IMU intervals");
    frames.push_back(
        {eigen(camera) * inverse(eigen(impl_->calibration.T_imu_rect)), timestamp, imu});
    while (frames.size() > 101)
        frames.pop_front();
    const double duration = (timestamp - frames.front().timestamp) * 1e-9;
    if (duration < 2 || frames.size() < 30)
        return std::nullopt;
    MX A(3 * (frames.size() - 1), 6);
    VX b(A.rows());
    V3 dv = V3::Zero(), dp = V3::Zero(), gyroError = V3::Zero();
    size_t row = 0;
    for (size_t i = 1; i < frames.size(); ++i) {
        const auto& previous = frames[i - 1];
        const auto& current = frames[i];
        const double frameDt = (current.timestamp - previous.timestamp) * 1e-9;
        const M3 R0 = previous.body.topLeftCorner<3, 3>(), R1 = current.body.topLeftCorner<3, 3>();
        const Eigen::Quaterniond q0(R0), q1(R1);
        V3 rotationIntegral = V3::Zero();
        for (size_t j = 1; j < current.imu.size(); ++j) {
            const auto& a = current.imu[j - 1];
            const auto& z = current.imu[j];
            const double dt = (z.timestampNs - a.timestampNs) * 1e-9;
            const double fraction =
                ((a.timestampNs - previous.timestamp) * 1e-9 + .5 * dt) / frameDt;
            const M3 R = q0.slerp(fraction, q1).toRotationMatrix();
            const V3 acc = R * (.5 * (eigen(a.acceleration) + eigen(z.acceleration)));
            dp += dv * dt + .5 * acc * dt * dt;
            dv += acc * dt;
            rotationIntegral += .5 * (eigen(a.angularVelocity) + eigen(z.angularVelocity)) * dt;
        }
        gyroError += rotationIntegral - log(R0.transpose() * R1);
        const double t = (current.timestamp - frames.front().timestamp) * 1e-9;
        A.block<3, 3>(row, 0) = M3::Identity() * t;
        A.block<3, 3>(row, 3) = M3::Identity() * .5 * t * t;
        b.segment<3>(row) =
            current.body.topRightCorner<3, 1>() - frames.front().body.topRightCorner<3, 1>() - dp;
        row += 3;
    }
    const VX solution = A.colPivHouseholderQr().solve(b);
    V3 gravity = solution.tail<3>();
    if (!solution.allFinite() || gravity.norm() < 8 || gravity.norm() > 11.5 ||
        gyroError.norm() / duration > .5)
        return std::nullopt;
    gravity = gravity.normalized() * 9.81;
    const V3 v0 = A.leftCols(3).colPivHouseholderQr().solve(b - A.rightCols(3) * gravity);
    if ((A.leftCols(3) * v0 + A.rightCols(3) * gravity - b).norm() /
            std::sqrt(double(frames.size() - 1)) >
        .15)
        return std::nullopt;
    InertialState result;
    result.initialized = true;
    result.T_world_rect = camera;
    result.velocity = cvvec(v0 + dv + gravity * duration);
    result.gravity = cvvec(gravity);
    result.gyroBias = cvvec(gyroError / duration);
    return result;
}
}
