#include "imuPreintegration.h"
#include "../inertialMath.h"
#include <stdexcept>
#include <algorithm>

namespace superVO::ba {
namespace {
struct Integral {
    M3 R = M3::Identity();
    V3 p = V3::Zero(), v = V3::Zero();
    M15 covariance = M15::Zero();
};
struct Noise {
    V3 gyro, accel;
};
// Linear interpolation has integrated curvature error approximately
// dt * second_difference / 12. Add its squared uncertainty separately from
// calibrated sensor noise, rather than treating all physical vibration as white
// measurement noise. Subtract the expected calibrated-noise contribution to
// the curvature statistic to avoid counting that noise a second time.
Noise integrationNoise(const std::vector<ImuSample>& samples, const ImuCalibration& c, bool model) {
    Noise result{V3::Constant(c.gyroNoise), V3::Constant(c.accelNoise)};
    if (!model || samples.size() < 7)
        return result;
    for (int sensor = 0; sensor < 2; ++sensor)
        for (int axis = 0; axis < 3; ++axis) {
            const double calibrated = sensor ? c.accelNoise : c.gyroNoise;
            double curvatureVariance = 0, whiteVariance = 0;
            for (size_t j = 1; j + 1 < samples.size(); ++j) {
                const double before = (samples[j].timestampNs - samples[j - 1].timestampNs) * 1e-9;
                const double after = (samples[j + 1].timestampNs - samples[j].timestampNs) * 1e-9;
                const double left = after / (before + after), right = before / (before + after);
                auto value = [&](size_t k) {
                    return sensor ? samples[k].acceleration[axis]
                                  : samples[k].angularVelocity[axis];
                };
                const double residual = value(j) - left * value(j - 1) - right * value(j + 1);
                // residual=-second_difference/2 at regular sample spacing.
                curvatureVariance += residual * residual * .5 * (before + after) / 36.;
                whiteVariance += calibrated * calibrated * (1 + left * left + right * right) / 36.;
            }
            const double extra =
                std::max(0., (curvatureVariance - whiteVariance) / double(samples.size() - 2));
            auto& density = sensor ? result.accel[axis] : result.gyro[axis];
            density = std::sqrt(calibrated * calibrated + extra);
        }
    return result;
}
Integral integrate(const std::vector<ImuSample>& s, const ImuCalibration& c, const V3& bg,
                   const V3& ba, const Noise* noise) {
    Integral out;
    for (size_t j = 1; j < s.size(); ++j) {
        const double dt = (s[j].timestampNs - s[j - 1].timestampNs) * 1e-9;
        const V3 w = .5 * (inertial::eigen(s[j - 1].angularVelocity) +
                           inertial::eigen(s[j].angularVelocity)) -
                     bg;
        const V3 a =
            .5 * (inertial::eigen(s[j - 1].acceleration) + inertial::eigen(s[j].acceleration)) - ba;
        const M3 half = inertial::exp(w * (.5 * dt)), Rmid = out.R * half;
        if (noise) {
            M15 F = M15::Identity(), Q = M15::Zero();
            F.block<3, 3>(0, 3) = M3::Identity() * dt;
            F.block<3, 3>(0, 6) = -.5 * out.R * inertial::skew(half * a) * dt * dt;
            F.block<3, 3>(3, 6) = -out.R * inertial::skew(half * a) * dt;
            F.block<3, 3>(0, 12) = -.5 * Rmid * dt * dt;
            F.block<3, 3>(3, 12) = -Rmid * dt;
            F.block<3, 3>(6, 6) = inertial::exp(-w * dt);
            const M3 W = inertial::skew(w * dt);
            const double angle = (w * dt).norm();
            M3 Jr = M3::Identity() - .5 * W + W * W / 6.;
            if (angle > 1e-5)
                Jr = M3::Identity() - (1 - std::cos(angle)) / (angle * angle) * W +
                     (angle - std::sin(angle)) / (angle * angle * angle) * W * W;
            F.block<3, 3>(6, 9) = -Jr * dt;
            F.block<3, 3>(3, 9) = .5 * Rmid * inertial::skew(a) * dt * dt;
            F.block<3, 3>(0, 9) = .25 * Rmid * inertial::skew(a) * dt * dt * dt;
            const M3 va =
                Rmid * noise->accel.array().square().matrix().asDiagonal() * Rmid.transpose();
            const M3 vg = noise->gyro.array().square().matrix().asDiagonal();
            Q.block<3, 3>(0, 0) = va * (dt * dt * dt / 3.);
            Q.block<3, 3>(0, 3) = Q.block<3, 3>(3, 0) = va * (dt * dt / 2.);
            Q.block<3, 3>(3, 3) = va * dt;
            Q.block<3, 3>(6, 6) = vg * dt;
            Q.block<3, 3>(9, 9) = M3::Identity() * (c.gyroWalk * c.gyroWalk * dt);
            Q.block<3, 3>(12, 12) = M3::Identity() * (c.accelWalk * c.accelWalk * dt);
            out.covariance = F * out.covariance * F.transpose() + Q;
        }
        out.p += out.v * dt + .5 * Rmid * a * dt * dt;
        out.v += Rmid * a * dt;
        out.R = out.R * inertial::exp(w * dt);
    }
    return out;
}
}
ImuPreintegration::ImuPreintegration(const std::vector<ImuSample>& s, const ImuCalibration& c,
                                     const V3& bg, const V3& ba, bool model)
    : gyroBias(bg), accelBias(ba), modelUncertainty(model), samples(s) {
    c.validate();
    if (s.size() < 2 || !bg.allFinite() || !ba.allFinite())
        throw std::invalid_argument("Invalid preintegration input");
    validateImuInterval(s, s.front().timestampNs, s.back().timestampNs, c.rateHz);
    dt = (s.back().timestampNs - s.front().timestampNs) * 1e-9;
    const auto noise = integrationNoise(s, c, model);
    gyroNoiseDensity = noise.gyro;
    accelNoiseDensity = noise.accel;
    auto base = integrate(s, c, bg, ba, &noise);
    rotation = Eigen::Quaterniond(base.R).normalized();
    position = base.p;
    velocity = base.v;
    covariance = .5 * (base.covariance + base.covariance.transpose());
    // A tiny roundoff floor, far below calibrated sensor variances.
    covariance.diagonal().array() += 1e-18;
    Eigen::LLT<M15> llt(covariance);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("IMU covariance is not positive definite");
    sqrtInformation = llt.matrixL().solve(M15::Identity());
    const double eps = 1e-5;
    for (int k = 0; k < 3; ++k) {
        V3 d = V3::Zero();
        d[k] = eps;
        const auto a = integrate(s, c, bg + d, ba, nullptr),
                   b = integrate(s, c, bg - d, ba, nullptr);
        rotationGyro.col(k) =
            (inertial::log(base.R.transpose() * a.R) - inertial::log(base.R.transpose() * b.R)) /
            (2 * eps);
        positionGyro.col(k) = (a.p - b.p) / (2 * eps);
        velocityGyro.col(k) = (a.v - b.v) / (2 * eps);
        const auto u = integrate(s, c, bg, ba + d, nullptr),
                   v = integrate(s, c, bg, ba - d, nullptr);
        positionAccel.col(k) = (u.p - v.p) / (2 * eps);
        velocityAccel.col(k) = (u.v - v.v) / (2 * eps);
    }
}
}
