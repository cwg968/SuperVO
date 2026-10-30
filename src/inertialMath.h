#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
namespace superVO::inertial {
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using M4 = Eigen::Matrix4d;
using MX = Eigen::MatrixXd;
using VX = Eigen::VectorXd;
inline V3 eigen(const cv::Vec3d& v) {
    return {v[0], v[1], v[2]};
}
inline cv::Vec3d cvvec(const V3& v) {
    return {v.x(), v.y(), v.z()};
}
inline M4 eigen(const cv::Matx44d& x) {
    M4 y;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            y(r, c) = x(r, c);
    return y;
}
inline cv::Matx44d cvpose(const M4& x) {
    cv::Matx44d y;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            y(r, c) = x(r, c);
    return y;
}
inline M3 skew(const V3& v) {
    M3 s;
    s << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
    return s;
}
inline M3 exp(const V3& v) {
    const double n = v.norm();
    if (n < 1e-10)
        return M3::Identity() + skew(v);
    return Eigen::AngleAxisd(n, v / n).toRotationMatrix();
}
inline V3 log(const M3& R) {
    Eigen::AngleAxisd aa(R);
    return aa.angle() * aa.axis();
}
inline M4 pose(const M3& R, const V3& p) {
    M4 T = M4::Identity();
    T.topLeftCorner<3, 3>() = R;
    T.topRightCorner<3, 1>() = p;
    return T;
}
inline M4 inverse(const M4& T) {
    M4 U = M4::Identity();
    U.topLeftCorner<3, 3>() = T.topLeftCorner<3, 3>().transpose();
    U.topRightCorner<3, 1>() = -U.topLeftCorner<3, 3>() * T.topRightCorner<3, 1>();
    return U;
}
}
