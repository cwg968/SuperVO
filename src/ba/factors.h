#pragma once
#include "imuPreintegration.h"
#include "gravityManifold.h"
#include "projectionFactor.h"
#include <ceres/ceres.h>
#include <ceres/autodiff_manifold.h>
#include <ceres/rotation.h>
#include <array>
#include <limits>
#include <optional>

namespace superVO::ba {
inline constexpr double zeroVelocitySigma = .03;
inline constexpr double zeroVelocityGate = 11.35; // approximately chi-square(3), 99%.
inline double zeroVelocityNis(const V3& velocity, const M3& covariance) {
    const M3 S = covariance + M3::Identity() * (zeroVelocitySigma * zeroVelocitySigma);
    Eigen::LLT<M3> factor(S);
    if (!velocity.allFinite() || !S.allFinite() || factor.info() != Eigen::Success)
        return std::numeric_limits<double>::infinity();
    return velocity.dot(factor.solve(velocity));
}
// Ambient layout q_xyzw, p, v, bg, ba; tangent layout right rotation, p,v,bg,ba.
using State = std::array<double, 16>;
template <class T> Eigen::Quaternion<T> quaternion(const T* x) {
    return {x[3], x[0], x[1], x[2]};
}
template <class T> Eigen::Matrix<T, 3, 1> vector(const T* x, int offset) {
    return Eigen::Map<const Eigen::Matrix<T, 3, 1>>(x + offset);
}
template <class T> Eigen::Quaternion<T> rotationExp(const Eigen::Matrix<T, 3, 1>& a) {
    T q[4];
    ceres::AngleAxisToQuaternion(a.data(), q);
    return {q[0], q[1], q[2], q[3]};
}
template <class T> Eigen::Matrix<T, 3, 1> rotationLog(const Eigen::Quaternion<T>& q) {
    const T a[4] = {q.w(), q.x(), q.y(), q.z()};
    Eigen::Matrix<T, 3, 1> r;
    ceres::QuaternionToAngleAxis(a, r.data());
    return r;
}
struct StateRetraction {
    template <class T> bool Plus(const T* x, const T* d, T* y) const {
        Eigen::Map<Eigen::Quaternion<T>> q(y);
        q = quaternion(x) * rotationExp<T>(vector(d, 0));
        for (int i = 4; i < 16; ++i)
            y[i] = x[i] + d[i - 1];
        return true;
    }
    template <class T> bool Minus(const T* y, const T* x, T* d) const {
        Eigen::Map<Eigen::Matrix<T, 3, 1>> r(d);
        r = rotationLog<T>(quaternion(x).conjugate() * quaternion(y));
        for (int i = 4; i < 16; ++i)
            d[i - 1] = y[i] - x[i];
        return true;
    }
};
using StateManifold = ceres::AutoDiffManifold<StateRetraction, 16, 15>;

struct ImuFactor {
    ImuPreintegration integral;
    template <class T> bool operator()(const T* a, const T* b, const T* g, T* out) const {
        const Eigen::Matrix<T, 3, 1> dbg = vector(a, 10) - integral.gyroBias.cast<T>(),
                                     dba = vector(a, 13) - integral.accelBias.cast<T>();
        const Eigen::Quaternion<T> dq =
            integral.rotation.cast<T>() *
            rotationExp<T>((integral.rotationGyro.cast<T>() * dbg).eval());
        const auto inv = quaternion(a).conjugate();
        const T dt(integral.dt);
        Eigen::Matrix<T, 15, 1> r;
        r.template segment<3>(0) = inv * (vector(b, 4) - vector(a, 4) - vector(a, 7) * dt -
                                          vector(g, 0) * (dt * dt / T(2))) -
                                   integral.position.cast<T>() -
                                   integral.positionGyro.cast<T>() * dbg -
                                   integral.positionAccel.cast<T>() * dba;
        r.template segment<3>(3) =
            inv * (vector(b, 7) - vector(a, 7) - vector(g, 0) * dt) - integral.velocity.cast<T>() -
            integral.velocityGyro.cast<T>() * dbg - integral.velocityAccel.cast<T>() * dba;
        r.template segment<3>(6) = rotationLog<T>(dq.conjugate() * inv * quaternion(b));
        r.template segment<3>(9) = vector(b, 10) - vector(a, 10);
        r.template segment<3>(12) = vector(b, 13) - vector(a, 13);
        Eigen::Map<Eigen::Matrix<T, 15, 1>> residual(out);
        residual = integral.sqrtInformation.cast<T>() * r;
        return true;
    }
};
struct VelocityFactor {
    template <class T> bool operator()(const T* x, T* out) const {
        for (int k = 0; k < 3; ++k)
            out[k] = x[k + 7] / T(zeroVelocitySigma);
        return true;
    }
};
// A fixed-linearization square-root prior with exact local-coordinate evaluation.
// Manifold definitions are shared by optimization, marginalization and covariance.
class PriorCost final : public ceres::CostFunction {
  public:
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::vector<State> anchors;
    std::optional<V3> gravityAnchor;
    PriorCost(Eigen::MatrixXd matrix, Eigen::VectorXd residual, std::vector<State> linearization,
              std::optional<V3> g = std::nullopt)
        : A(std::move(matrix)), b(std::move(residual)), anchors(std::move(linearization)),
          gravityAnchor(g) {
        set_num_residuals(A.rows());
        for (size_t i = 0; i < anchors.size(); ++i)
            mutable_parameter_block_sizes()->push_back(16);
        if (gravityAnchor)
            mutable_parameter_block_sizes()->push_back(3);
    }
    bool Evaluate(double const* const* x, double* residual, double** jacobians) const override {
        Eigen::VectorXd delta(A.cols());
        StateRetraction retract;
        for (size_t k = 0; k < anchors.size(); ++k)
            retract.Minus(x[k], anchors[k].data(), delta.data() + 15 * k);
        if (gravityAnchor && !GravityRetraction{}.Minus(x[anchors.size()], gravityAnchor->data(),
                                                        delta.data() + delta.size() - 2))
            return false;
        Eigen::Map<Eigen::VectorXd> r(residual, A.rows());
        r = A * delta + b;
        if (jacobians)
            for (size_t k = 0; k < anchors.size(); ++k)
                if (jacobians[k]) {
                    minusJacobian(x[k], anchors[k], jacobians[k], k);
                }
        if (gravityAnchor && jacobians && jacobians[anchors.size()]) {
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>> J(
                jacobians[anchors.size()], A.rows(), 3);
            ceres::AutoDiffCostFunction<GravityMinusFactor, 2, 3> cost(
                new GravityMinusFactor{*gravityAnchor});
            Eigen::Matrix<double, 2, 3, Eigen::RowMajor> local;
            double residual[2];
            const double* inputs[] = {x[anchors.size()]};
            double* jac[] = {local.data()};
            if (!cost.Evaluate(inputs, residual, jac))
                return false;
            J = A.rightCols(2) * local;
        }
        return r.allFinite();
    }

  private:
    void minusJacobian(const double* x, const State& anchor, double* output, size_t index) const;
};
struct MinusFactor {
    State anchor;
    template <class T> bool operator()(const T* p, T* d) const {
        T a[16];
        for (int j = 0; j < 16; ++j)
            a[j] = T(anchor[j]);
        return StateRetraction{}.Minus(p, a, d);
    }
};
inline void PriorCost::minusJacobian(const double* x, const State& anchor, double* output,
                                     size_t index) const {
    ceres::AutoDiffCostFunction<MinusFactor, 15, 16> cost(new MinusFactor{anchor});
    Eigen::Matrix<double, 15, 16, Eigen::RowMajor> J;
    double r[15];
    const double* inputs[] = {x};
    double* jacobians[] = {J.data()};
    cost.Evaluate(inputs, r, jacobians);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 16, Eigen::RowMajor>> out(output, A.rows(),
                                                                               16);
    out = A.middleCols(15 * index, 15) * J;
}
}
