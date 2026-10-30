#include <poseEstimation.h>
#include <opencv2/calib3d.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace superVO {
namespace {
using Matrix6 = cv::Matx<double, 6, 6>;
using Vector6 = cv::Vec<double, 6>;
struct Measurement {
    size_t id;
    int camera;
    cv::Point2d pixel;
    double weight;
};
bool rigid(const cv::Matx44d& T) {
    if (!cv::checkRange(cv::Mat(T)))
        return false;
    cv::Matx33d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = T(r, c);
    return cv::norm(cv::Mat(R.t() * R - cv::Matx33d::eye())) < 1e-4 &&
           std::abs(cv::determinant(R) - 1) < 1e-4 &&
           cv::norm(cv::Vec4d(T(3, 0), T(3, 1), T(3, 2), T(3, 3) - 1)) < 1e-8;
}
cv::Matx44d increment(const cv::Matx44d& T, const Vector6& dx) {
    cv::Matx33d R;
    cv::Rodrigues(cv::Vec3d(dx[0], dx[1], dx[2]), R);
    auto D = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            D(r, c) = R(r, c);
        D(r, 3) = dx[r + 3];
    }
    return D * T;
}
bool project(const cv::Point3f& p, const cv::Matx44d& T, const cv::Matx44d& E, int camera,
             const cv::Mat& K, cv::Point2d& pixel, cv::Matx<double, 2, 6>& J) {
    const auto q = T * cv::Vec4d(p.x, p.y, p.z, 1), v = camera ? E * q : q;
    if (v[2] <= 1e-6 || !std::isfinite(v[0]) || !std::isfinite(v[1]) || !std::isfinite(v[2]))
        return false;
    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), iz = 1 / v[2];
    pixel = {fx * v[0] * iz + K.at<double>(0, 2), fy * v[1] * iz + K.at<double>(1, 2)};
    cv::Matx33d R = cv::Matx33d::eye();
    if (camera)
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R(r, c) = E(r, c);
    const cv::Matx<double, 2, 3> projection(fx * iz, 0, -fx * v[0] * iz * iz, 0, fy * iz,
                                            -fy * v[1] * iz * iz);
    const cv::Matx<double, 3, 6> motion{0, q[2], -q[1], 1,    0,     0, -q[2], 0, q[0],
                                        0, 1,    0,     q[1], -q[0], 0, 0,     0, 1};
    J = projection * R * motion;
    return std::isfinite(pixel.x) && std::isfinite(pixel.y);
}
double linearize(const cv::Matx44d& T, const std::vector<cv::Point3f>& points,
                 const std::vector<Measurement>& observations, const cv::Mat& K,
                 const cv::Matx44d& E, double threshold, Matrix6* H = nullptr, Vector6* g = nullptr,
                 bool inliersOnly = false) {
    if (H)
        *H = Matrix6::zeros();
    if (g)
        *g = Vector6::all(0);
    double cost = 0;
    for (const auto& m : observations) {
        cv::Point2d projected;
        cv::Matx<double, 2, 6> J;
        if (!project(points[m.id], T, E, m.camera, K, projected, J)) {
            cost += 1e6;
            continue;
        }
        const cv::Vec2d residual(m.pixel.x - projected.x, m.pixel.y - projected.y);
        const double error = cv::norm(residual);
        if (inliersOnly && error > threshold)
            continue;
        cost += m.weight * (error <= threshold ? error * error
                                               : 2 * threshold * error - threshold * threshold);
        const double weight = m.weight * (error <= threshold ? 1 : threshold / error);
        if (H)
            *H += weight * (J.t() * J);
        if (g)
            *g += weight * (J.t() * residual);
    }
    return cost;
}
cv::Matx44d optimize(cv::Matx44d T, const std::vector<cv::Point3f>& points,
                     const std::vector<Measurement>& measurements, const cv::Mat& K,
                     const cv::Matx44d& E, double threshold) {
    double damping = 1e-3;
    for (int iteration = 0; iteration < 30; ++iteration) {
        Matrix6 H;
        Vector6 g, dx;
        const double cost = linearize(T, points, measurements, K, E, threshold, &H, &g);
        for (int j = 0; j < 6; ++j)
            H(j, j) += damping * std::max(1., H(j, j));
        if (!cv::solve(H, g, dx, cv::DECOMP_CHOLESKY) || !cv::checkRange(cv::Mat(dx)))
            break;
        const auto next = increment(T, dx);
        if (linearize(next, points, measurements, K, E, threshold) < cost) {
            T = next;
            damping = std::max(1e-8, damping * .3);
            if (cv::norm(dx) < 1e-7)
                break;
        } else {
            damping *= 10;
            if (damping > 1e8)
                break;
        }
    }
    return T;
}
double conditioning(Matrix6 H) {
    Vector6 scales;
    for (int j = 0; j < 6; ++j) {
        if (H(j, j) <= 1e-12 || !std::isfinite(H(j, j)))
            return 0;
        scales[j] = std::sqrt(H(j, j));
    }
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c)
            H(r, c) /= scales[r] * scales[c];
    cv::Mat eigenvalues;
    if (!cv::eigen(cv::Mat(H), eigenvalues))
        return 0;
    return std::max(0., eigenvalues.at<double>(5) / eigenvalues.at<double>(0));
}
}
StereoPoseEstimate estimateStereoPose(const std::vector<cv::Point3f>& points,
                                      const std::array<PointTracks, 2>& tracks, const cv::Mat& K,
                                      const cv::Matx44d& E, cv::Size size,
                                      const cv::Matx44d& initial, int minimum, double threshold) {
    StereoPoseEstimate best;
    if (minimum < 6 || threshold <= 0 || !std::isfinite(threshold) || size.width <= 0 ||
        size.height <= 0 || K.type() != CV_64F || K.rows != 3 || K.cols != 3 ||
        !cv::checkRange(K) || K.at<double>(0, 0) <= 0 || K.at<double>(1, 1) <= 0 || !rigid(E) ||
        !rigid(initial)) {
        best.rejection = 6;
        return best;
    }
    std::vector<int> counts(points.size(), 0);
    for (int c = 0; c < 2; ++c) {
        if (tracks[c].pixels.size() != tracks[c].sourceIndices.size())
            throw std::invalid_argument("Invalid stereo measurement array");
        std::vector<bool> seen(points.size(), false);
        for (size_t j = 0; j < tracks[c].pixels.size(); ++j) {
            const size_t id = tracks[c].sourceIndices[j];
            if (id >= points.size() || seen[id])
                throw std::invalid_argument("Invalid or duplicate stereo landmark ID");
            seen[id] = true;
            const auto& p = points[id];
            const auto pixel = tracks[c].pixels[j];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
                !std::isfinite(pixel.x) || !std::isfinite(pixel.y) || pixel.x < 0 || pixel.y < 0 ||
                pixel.x >= size.width || pixel.y >= size.height) {
                best.rejection = 6;
                return best;
            }
            ++counts[id];
        }
    }
    best.uniqueObservations =
        std::count_if(counts.begin(), counts.end(), [](int n) { return n > 0; });
    if (best.uniqueObservations < static_cast<size_t>(minimum))
        return best;
    std::vector<Measurement> measurements;
    for (int c = 0; c < 2; ++c)
        for (size_t j = 0; j < tracks[c].pixels.size(); ++j) {
            auto id = tracks[c].sourceIndices[j];
            measurements.push_back({id, c, tracks[c].pixels[j], 1. / counts[id]});
        }
    std::vector<cv::Matx44d> seeds{initial};
    for (int c = 0; c < 2; ++c) {
        std::vector<cv::Point3f> p;
        for (auto id : tracks[c].sourceIndices)
            p.push_back(points[id]);
        const auto state = cv::theRNG().state;
        const auto seed = estimatePose(p, tracks[c].pixels, K, 6, threshold);
        cv::theRNG().state = state;
        if (seed.valid)
            seeds.push_back(c ? leftPoseFromRight(seed.T_camera_reference, E)
                              : seed.T_camera_reference);
    }
    best.rejection = 2;
    for (const auto& seed : seeds) {
        const auto pose = optimize(seed, points, measurements, K, E, threshold);
        const auto quality = assessStereoPose(pose, points, tracks, K, E, size, threshold);
        int rejection = 0;
        if (quality.uniqueInliers < static_cast<size_t>(minimum) ||
            quality.uniqueInliers < .4 * best.uniqueObservations)
            rejection = 3;
        else if (quality.coverage < .25)
            rejection = 4;
        Matrix6 H;
        linearize(pose, points, measurements, K, E, threshold, &H, nullptr, true);
        const double condition = conditioning(H);
        if (!rejection && condition < 1e-6)
            rejection = 5;
        if (rejection) {
            if (!best.pose.valid)
                best.rejection = rejection;
            continue;
        }
        if (best.pose.valid && quality.score <= best.quality.score)
            continue;
        best.pose.valid = true;
        best.pose.T_camera_reference = pose;
        best.pose.rmse = quality.rmse;
        best.quality = quality;
        best.conditionRatio = condition;
        best.rejection = 0;
        best.pose.inliers.clear();
        std::vector<bool> supported(points.size(), false);
        for (int c = 0; c < 2; ++c)
            for (auto j : quality.inlierObservations[c])
                supported[tracks[c].sourceIndices[j]] = true;
        for (size_t j = 0; j < supported.size(); ++j)
            if (supported[j])
                best.pose.inliers.push_back(int(j));
    }
    return best;
}
}
