#include <stereoBaEstimator.h>
#include "projectionFactor.h"
#include <ceres/ceres.h>
#include <ceres/product_manifold.h>
#include <cmath>
#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <stdexcept>

namespace superVO {
namespace {
using Pose = std::array<double, 7>;
using Point = std::array<double, 3>;
Pose packed(const cv::Matx44d& T) {
    Eigen::Matrix3d R;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            R(r, c) = T(r, c);
    if (!cv::checkRange(cv::Mat(T)) ||
        (R.transpose() * R - Eigen::Matrix3d::Identity()).norm() > 1e-6 ||
        std::abs(R.determinant() - 1) > 1e-6 ||
        cv::norm(cv::Vec4d(T(3, 0), T(3, 1), T(3, 2), T(3, 3) - 1)) > 1e-9)
        throw std::invalid_argument(
            "BA requires a finite rigid camera pose; orthogonality=" +
            std::to_string((R.transpose() * R - Eigen::Matrix3d::Identity()).norm()) +
            ", determinant=" + std::to_string(R.determinant()));
    const Eigen::Quaterniond q = Eigen::Quaterniond(R).normalized();
    return {q.x(), q.y(), q.z(), q.w(), T(0, 3), T(1, 3), T(2, 3)};
}
cv::Matx44d unpacked(const Pose& x) {
    const Eigen::Matrix3d R = Eigen::Quaterniond(x[3], x[0], x[1], x[2]).toRotationMatrix();
    auto T = cv::Matx44d::eye();
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            T(r, c) = R(r, c);
        T(r, 3) = x[r + 4];
    }
    return T;
}
}
struct StereoBaEstimator::Impl {
    struct Node {
        Pose pose;
        VisualMeasurements measurements;
        bool keyframe;
    };
    StereoCalibration camera;
    StereoBaOptions options;
    std::map<int, Node> nodes;
    std::map<LandmarkId, Point> geometry;
    StereoBaDiagnostics diagnostic;
    Impl(StereoCalibration c, StereoBaOptions o) : camera(std::move(c)), options(o) {
        camera.validate();
        camera.intrinsics = camera.intrinsics.clone();
        if (o.keyframes < 2 || o.keyframes > 12 || o.landmarks < o.minInliers ||
            o.landmarks > 2000 || o.iterations < 1 || o.iterations > 30 || o.minInliers < 6)
            throw std::invalid_argument("Invalid stereo BA configuration");
    }
    ba::ProjectionFactor projection(const PixelMeasurement& p) const {
        Eigen::Matrix2d C;
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 2; ++c)
                C(r, c) = p.covariance(r, c);
        Eigen::LLT<Eigen::Matrix2d> factor(C);
        if (!C.allFinite() || (C - C.transpose()).norm() > 1e-9 || factor.info() != Eigen::Success)
            throw std::invalid_argument("Invalid stereo BA pixel covariance");
        return {Eigen::Matrix3d::Identity(),
                Eigen::Vector3d(p.camera ? -camera.baseline : 0, 0, 0),
                factor.matrixL().solve(Eigen::Matrix2d::Identity()),
                {p.pixel.x, p.pixel.y},
                camera.intrinsics.at<double>(0, 0),
                camera.intrinsics.at<double>(1, 1),
                camera.intrinsics.at<double>(0, 2),
                camera.intrinsics.at<double>(1, 2)};
    }
    void trim() {
        while (int(nodes.size()) > options.keyframes)
            nodes.erase(nodes.begin());
        std::set<LandmarkId> live;
        for (const auto& [frame, n] : nodes)
            for (const auto& p : n.measurements.pixels)
                live.insert(p.landmark);
        for (auto it = geometry.begin(); it != geometry.end();) {
            if (!live.count(it->first))
                it = geometry.erase(it);
            else
                ++it;
        }
    }
    bool solve() {
        struct Support {
            std::map<int, int> cameras;
            int cell = 0;
        };
        std::map<LandmarkId, Support> support;
        const int current = nodes.rbegin()->first;
        for (const auto& [frame, n] : nodes)
            for (const auto& p : n.measurements.pixels) {
                auto& s = support[p.landmark];
                s.cameras[frame] |= 1 << p.camera;
                s.cell = int(p.pixel.y * 6 / camera.imageSize.height) * 8 +
                         int(p.pixel.x * 8 / camera.imageSize.width);
            }
        std::array<std::vector<LandmarkId>, 48> cells;
        for (const auto& [id, s] : support) {
            bool stereo = false;
            for (const auto& [frame, mask] : s.cameras)
                stereo |= mask == 3;
            if (s.cameras.size() >= 2 && stereo && geometry.count(id))
                cells.at(s.cell).push_back(id);
        }
        for (auto& cell : cells)
            std::stable_sort(cell.begin(), cell.end(), [&](auto a, auto b) {
                const auto& x = support.at(a).cameras;
                const auto& y = support.at(b).cameras;
                if (x.count(current) != y.count(current))
                    return x.count(current) > y.count(current);
                return x.size() > y.size();
            });
        std::map<LandmarkId, Point> selected;
        // Reserve bridge tracks before filling the spatial budget. Otherwise a
        // large fresh point set can evict the only edges connecting old poses.
        for (auto node = nodes.rbegin(); node != nodes.rend(); ++node) {
            const int frame = node->first;
            if (frame == nodes.begin()->first)
                continue;
            int reserved = 0;
            for (size_t round = 0;
                 reserved < options.minInliers && selected.size() < size_t(options.landmarks);
                 ++round) {
                bool any = false;
                for (const auto& cell : cells)
                    if (round < cell.size()) {
                        any = true;
                        const auto id = cell[round];
                        const auto& frames = support.at(id).cameras;
                        if (frames.count(frame) && frames.begin()->first < frame) {
                            selected.emplace(id, geometry.at(id));
                            if (++reserved >= options.minInliers ||
                                selected.size() >= size_t(options.landmarks))
                                break;
                        }
                    }
                if (!any)
                    break;
            }
        }
        for (size_t round = 0; selected.size() < size_t(options.landmarks); ++round) {
            bool any = false;
            for (const auto& cell : cells)
                if (round < cell.size() && selected.size() < size_t(options.landmarks)) {
                    const auto id = cell[round];
                    selected.emplace(id, geometry.at(id));
                    any = true;
                }
            if (!any)
                break;
        }
        diagnostic.landmarks = selected.size();
        std::map<int, int> counts;
        std::set<int> connected{nodes.begin()->first};
        for (const auto& [id, x] : selected)
            for (const auto& [f, mask] : support.at(id).cameras)
                ++counts[f];
        for (size_t round = 0; round < nodes.size(); ++round)
            for (const auto& [id, x] : selected) {
                const auto& frames = support.at(id).cameras;
                if (std::any_of(frames.begin(), frames.end(),
                                [&](const auto& f) { return connected.count(f.first); }))
                    for (const auto& [f, mask] : frames)
                        connected.insert(f);
            }
        if (counts[current] < options.minInliers || !connected.count(current)) {
            diagnostic.rejection = 2;
            return false;
        }
        std::map<int, cv::Matx44d> before;
        for (const auto& [f, n] : nodes)
            before.emplace(f, unpacked(n.pose));
        ceres::Problem problem;
        for (auto& [frame, n] : nodes) {
            using Manifold =
                ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;
            problem.AddParameterBlock(n.pose.data(), 7, new Manifold);
            if (frame == nodes.begin()->first || counts[frame] < 6 || !connected.count(frame))
                problem.SetParameterBlockConstant(n.pose.data());
        }
        auto ordering = std::make_shared<ceres::ParameterBlockOrdering>();
        for (auto& [id, x] : selected)
            ordering->AddElementToGroup(x.data(), 0);
        for (auto& [frame, n] : nodes) {
            ordering->AddElementToGroup(n.pose.data(), 1);
            for (const auto& p : n.measurements.pixels)
                if (auto it = selected.find(p.landmark); it != selected.end()) {
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<ba::ProjectionFactor, 2, 7, 3>(
                            new ba::ProjectionFactor(projection(p))),
                        new ceres::HuberLoss(2.5), n.pose.data(), it->second.data());
                    ++diagnostic.observations;
                }
        }
        ceres::Solver::Options settings;
        settings.linear_solver_type = ceres::DENSE_SCHUR;
        settings.linear_solver_ordering = ordering;
        settings.max_num_iterations = options.iterations;
        settings.max_solver_time_in_seconds = .03;
        settings.num_threads = 1;
        settings.function_tolerance = 1e-5;
        ceres::Solver::Summary summary;
        diagnostic.attempted = true;
        ceres::Solve(settings, &problem, &summary);
        diagnostic.initialCost = summary.initial_cost;
        diagnostic.finalCost = summary.final_cost;
        diagnostic.iterations = summary.iterations.size();
        if (!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost) ||
            summary.final_cost > summary.initial_cost + 1e-6) {
            diagnostic.rejection = 3;
            return false;
        }
        for (const auto& [f, n] : nodes)
            for (double x : n.pose)
                if (!std::isfinite(x)) {
                    diagnostic.rejection = 3;
                    return false;
                }
        std::set<LandmarkId> inliers;
        std::set<int> coverage;
        for (const auto& p : nodes.at(current).measurements.pixels)
            if (auto it = selected.find(p.landmark); it != selected.end()) {
                double residual[2];
                auto unweighted = p;
                unweighted.covariance = cv::Matx22d::eye();
                if (projection(unweighted)(nodes.at(current).pose.data(), it->second.data(),
                                           residual) &&
                    std::hypot(residual[0], residual[1]) < 3) {
                    inliers.insert(p.landmark);
                    coverage.insert(int(p.pixel.y * 6 / camera.imageSize.height) * 8 +
                                    int(p.pixel.x * 8 / camera.imageSize.width));
                }
            }
        diagnostic.inliers = inliers.size();
        if (inliers.size() < size_t(options.minInliers) || inliers.size() < .6 * counts[current] ||
            coverage.size() < 4) {
            diagnostic.rejection = 4;
            return false;
        }
        // Unoptimized seeds move with an observing anchor, including fresh points
        // first measured in this keyframe. They are not independent 3D priors.
        std::map<int, cv::Matx44d> corrections;
        for (const auto& [f, n] : nodes)
            corrections.emplace(f, unpacked(n.pose) * before.at(f).inv(cv::DECOMP_SVD));
        for (auto& [id, x] : geometry)
            if (!selected.count(id)) {
                const auto host = support.at(id).cameras.begin()->first;
                const auto q = corrections.at(host) * cv::Vec4d(x[0], x[1], x[2], 1);
                x = {q[0], q[1], q[2]};
            }
        for (const auto& [id, x] : selected) {
            if (!std::all_of(x.begin(), x.end(), [](double v) { return std::isfinite(v); })) {
                diagnostic.rejection = 3;
                return false;
            }
            geometry.at(id) = x;
        }
        diagnostic.accepted = true;
        diagnostic.rejection = 0;
        return true;
    }
};
StereoBaEstimator::StereoBaEstimator(StereoCalibration c, StereoBaOptions o)
    : impl_(std::make_unique<Impl>(std::move(c), o)) {}
StereoBaEstimator::~StereoBaEstimator() = default;
StereoBaEstimator::StereoBaEstimator(const StereoBaEstimator& x)
    : impl_(std::make_unique<Impl>(*x.impl_)) {}
StereoBaEstimator& StereoBaEstimator::operator=(const StereoBaEstimator& x) {
    if (this != &x)
        impl_ = std::make_unique<Impl>(*x.impl_);
    return *this;
}
bool StereoBaEstimator::update(const VisualMeasurements& batch, const cv::Matx44d& initial,
                               bool keyframe) {
    auto& w = *impl_;
    const auto pose = packed(initial);
    if (batch.frame < 0 || batch.timestamp < 0 ||
        (!w.nodes.empty() && (batch.frame <= w.nodes.rbegin()->first ||
                              batch.timestamp <= w.nodes.rbegin()->second.measurements.timestamp)))
        throw std::invalid_argument("Stereo BA frames must increase");
    if (batch.pixels.size() > 4000 || batch.seeds.size() > 2000)
        throw std::invalid_argument("Stereo BA observation budget exceeded");
    std::set<std::pair<LandmarkId, int>> unique;
    std::set<LandmarkId> seeds;
    for (const auto& s : batch.seeds)
        if (!s.landmark || !seeds.insert(s.landmark).second || !std::isfinite(s.world.x) ||
            !std::isfinite(s.world.y) || !std::isfinite(s.world.z))
            throw std::invalid_argument("Invalid stereo BA seed");
    for (const auto& p : batch.pixels) {
        if (!p.landmark || p.camera < 0 || p.camera > 1 || !std::isfinite(p.pixel.x) ||
            !std::isfinite(p.pixel.y) || p.pixel.x < 0 || p.pixel.y < 0 ||
            p.pixel.x >= w.camera.imageSize.width || p.pixel.y >= w.camera.imageSize.height ||
            !unique.emplace(p.landmark, p.camera).second ||
            (!seeds.count(p.landmark) && !w.geometry.count(p.landmark)))
            throw std::invalid_argument("Invalid/duplicate stereo BA pixel");
        w.projection(p);
    }
    if (!w.nodes.empty() && !w.nodes.rbegin()->second.keyframe)
        w.nodes.erase(std::prev(w.nodes.end()));
    w.nodes.emplace(batch.frame, Impl::Node{pose, batch, keyframe});
    for (const auto& s : batch.seeds)
        w.geometry.try_emplace(s.landmark, Point{s.world.x, s.world.y, s.world.z});
    w.trim();
    w.diagnostic = {};
    w.diagnostic.states = w.nodes.size();
    if (!keyframe || w.nodes.size() < 2)
        return false;
    // Solve on a transaction; rejected results never change accepted PnP geometry.
    auto candidate = w;
    if (candidate.solve()) {
        w = std::move(candidate);
        return true;
    }
    w.diagnostic = candidate.diagnostic;
    return false;
}
std::optional<cv::Matx44d> StereoBaEstimator::pose(int frame) const {
    const auto it = impl_->nodes.find(frame);
    return it == impl_->nodes.end() ? std::nullopt
                                    : std::optional<cv::Matx44d>(unpacked(it->second.pose));
}
std::vector<BaPoseUpdate> StereoBaEstimator::poses() const {
    std::vector<BaPoseUpdate> out;
    for (const auto& [f, n] : impl_->nodes)
        out.push_back({f, unpacked(n.pose)});
    return out;
}
std::vector<LandmarkSeed> StereoBaEstimator::landmarks() const {
    std::vector<LandmarkSeed> out;
    for (const auto& [id, x] : impl_->geometry)
        out.push_back({id, {x[0], x[1], x[2]}});
    return out;
}
StereoBaDiagnostics StereoBaEstimator::diagnostics() const {
    return impl_->diagnostic;
}
}
