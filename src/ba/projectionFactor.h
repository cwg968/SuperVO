#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace superVO::ba {
// Both pose-only and inertial states start with q_xyzw, world translation.
// The fixed transform maps this state's body/camera coordinates to the observing camera.
struct ProjectionFactor {
    Eigen::Matrix3d R_camera_imu;
    Eigen::Vector3d t_camera_imu;
    Eigen::Matrix2d W;
    Eigen::Vector2d pixel;
    double fx, fy, cx, cy;
    template <class T> bool operator()(const T* s, const T* point, T* out) const {
        const Eigen::Quaternion<T> q(s[3], s[0], s[1], s[2]);
        const Eigen::Matrix<T, 3, 1> body =
            q.conjugate() * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point) -
                             Eigen::Map<const Eigen::Matrix<T, 3, 1>>(s + 4));
        const Eigen::Matrix<T, 3, 1> p = R_camera_imu.cast<T>() * body + t_camera_imu.cast<T>();
        if (p.z() <= T(.02))
            return false;
        const Eigen::Matrix<T, 2, 1> e(T(fx) * p.x() / p.z() + T(cx) - T(pixel.x()),
                                       T(fy) * p.y() / p.z() + T(cy) - T(pixel.y()));
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residual(out);
        residual = W.cast<T>() * e;
        return true;
    }
};
}
