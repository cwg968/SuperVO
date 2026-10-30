#pragma once
#include "imuPreintegration.h"
#include <ceres/autodiff_manifold.h>

namespace superVO::ba {
inline constexpr double gravityMagnitude = 9.81;

// Gravity has known magnitude and two angular degrees of freedom. All gravity
// priors, marginalization columns and recovered covariances use these radians.
struct GravityRetraction {
    template <class T> static Eigen::Matrix<T, 3, 2> basis(const T* x) {
        const Eigen::Matrix<T, 3, 1> n =
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(x) / T(gravityMagnitude);
        int axis = 0;
        for (int k = 1; k < 3; ++k)
            if (ceres::abs(n[k]) < ceres::abs(n[axis]))
                axis = k;
        Eigen::Matrix<T, 3, 1> a = Eigen::Matrix<T, 3, 1>::Zero();
        a[axis] = T(1);
        Eigen::Matrix<T, 3, 2> B;
        B.col(0) = (a - n * n[axis]).normalized();
        B.col(1) = n.cross(B.col(0));
        return B;
    }
    template <class T> bool Plus(const T* x, const T* delta, T* y) const {
        const Eigen::Map<const Eigen::Matrix<T, 2, 1>> d(delta);
        const T theta2 = d.squaredNorm();
        T cosine, sinc;
        if (theta2 > T(1e-12)) {
            const T theta = ceres::sqrt(theta2);
            cosine = ceres::cos(theta);
            sinc = ceres::sin(theta) / theta;
        } else {
            cosine = T(1) - theta2 / T(2);
            sinc = T(1) - theta2 / T(6);
        }
        Eigen::Map<Eigen::Matrix<T, 3, 1>> out(y);
        out = cosine * Eigen::Map<const Eigen::Matrix<T, 3, 1>>(x) +
              T(gravityMagnitude) * sinc * basis(x) * d;
        return true;
    }
    template <class T> bool Minus(const T* y, const T* x, T* delta) const {
        const Eigen::Matrix<T, 3, 1> n =
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(x) / T(gravityMagnitude);
        const Eigen::Matrix<T, 3, 1> m =
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(y) / T(gravityMagnitude);
        const T cosine = n.dot(m);
        const Eigen::Matrix<T, 3, 1> u = m - cosine * n;
        const T sine2 = u.squaredNorm();
        T scale;
        if (sine2 > T(1e-12)) {
            const T sine = ceres::sqrt(sine2);
            scale = ceres::atan2(sine, cosine) / sine;
        } else {
            if (cosine < T(0))
                return false; // Antipodal directions have no unique local difference.
            scale = T(1) + sine2 / T(6);
        }
        Eigen::Map<Eigen::Matrix<T, 2, 1>> out(delta);
        out = scale * basis(x).transpose() * u;
        return true;
    }
};
using GravityManifold = ceres::AutoDiffManifold<GravityRetraction, 3, 2>;
struct GravityMinusFactor {
    V3 anchor;
    template <class T> bool operator()(const T* g, T* delta) const {
        const Eigen::Matrix<T, 3, 1> a = anchor.cast<T>();
        return GravityRetraction{}.Minus(g, a.data(), delta);
    }
};
}
