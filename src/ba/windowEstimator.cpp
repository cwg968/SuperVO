#include "windowInternal.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace superVO {
using namespace inertial;
namespace {
ba::State stateVector(const InertialState& s, const Eigen::Matrix4d& bodyFromCamera) {
    const M4 T = eigen(s.T_world_rect) * inverse(bodyFromCamera);
    ba::State x{};
    Eigen::Map<Eigen::Quaterniond> q(x.data());
    q = Eigen::Quaterniond(T.topLeftCorner<3, 3>()).normalized();
    Eigen::Map<V3>(x.data() + 4) = T.topRightCorner<3, 1>();
    Eigen::Map<V3>(x.data() + 7) = eigen(s.velocity);
    Eigen::Map<V3>(x.data() + 10) = eigen(s.gyroBias);
    Eigen::Map<V3>(x.data() + 13) = eigen(s.accelBias);
    return x;
}
bool finite(const ba::State& x) {
    return Eigen::Map<const Eigen::Matrix<double, 16, 1>>(x.data()).allFinite() &&
           std::abs(ba::quaternion(x.data()).norm() - 1) < 1e-6 &&
           ba::vector(x.data(), 7).norm() < 30 && ba::vector(x.data(), 10).norm() < 1 &&
           ba::vector(x.data(), 13).norm() < 5;
}
}
WindowEstimator::Impl::Impl(StereoCalibration c, ImuCalibration i, WindowOptions o)
    : camera(std::move(c)), imu(i), options(o), bodyFromCamera(eigen(i.T_imu_rect)) {
    camera.validate();
    imu.validate();
    if (options.states < 8 || options.states > 20 || options.landmarks < options.minInliers ||
        options.landmarks > 2000 || options.iterations < 1 || options.iterations > 30 ||
        options.minInliers < 6 || (options.resolveAfterPruning && !options.observationFeedback) ||
        options.auditEvery < 0)
        throw std::invalid_argument("Invalid sliding-window configuration");
    camera.intrinsics = camera.intrinsics.clone();
}
cv::Matx44d WindowEstimator::Impl::pose(const ba::State& x) const {
    return cvpose(
        inertial::pose(ba::quaternion(x.data()).toRotationMatrix(), ba::vector(x.data(), 4)) *
        bodyFromCamera);
}
ba::ProjectionFactor WindowEstimator::Impl::projection(const PixelMeasurement& p) const {
    const M4 E =
        eigen(p.camera ? camera.rightFromLeft() : cv::Matx44d::eye()) * inverse(bodyFromCamera);
    Eigen::Matrix2d C;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            C(r, c) = p.covariance(r, c);
    Eigen::LLT<Eigen::Matrix2d> factor(C);
    if (!C.allFinite() || (C - C.transpose()).norm() > 1e-9 || factor.info() != Eigen::Success)
        throw std::invalid_argument(
            "Pixel covariance must be finite, symmetric and positive definite");
    return {E.topLeftCorner<3, 3>(),
            E.topRightCorner<3, 1>(),
            factor.matrixL().solve(Eigen::Matrix2d::Identity()),
            {p.pixel.x, p.pixel.y},
            camera.intrinsics.at<double>(0, 0),
            camera.intrinsics.at<double>(1, 1),
            camera.intrinsics.at<double>(0, 2),
            camera.intrinsics.at<double>(1, 2)};
}
void WindowEstimator::Impl::add(const VisualMeasurements& batch) {
    if (batch.frame != current().frame || batch.timestamp != current().timestamp)
        throw std::invalid_argument("Visual packet must match the current IMU state");
    std::map<LandmarkId, cv::Point3d> seeds;
    for (const auto& s : batch.seeds) {
        if (!s.landmark || !std::isfinite(s.world.x) || !std::isfinite(s.world.y) ||
            !std::isfinite(s.world.z) || !seeds.emplace(s.landmark, s.world).second)
            throw std::invalid_argument("Invalid landmark seed");
    }
    // Round-robin image cells prevents an ID/score ordering from filling the
    // optimization budget with a small high-texture region.
    std::array<std::vector<LandmarkId>, 48> cells;
    std::set<LandmarkId> candidates;
    std::set<std::pair<LandmarkId, int>> unique;
    for (const auto& p : batch.pixels) {
        if (!p.landmark || p.camera < 0 || p.camera > 1 || !std::isfinite(p.pixel.x) ||
            !std::isfinite(p.pixel.y) || p.pixel.x < 0 || p.pixel.y < 0 ||
            p.pixel.x >= camera.imageSize.width || p.pixel.y >= camera.imageSize.height ||
            !unique.insert({p.landmark, p.camera}).second)
            throw std::invalid_argument("Invalid/duplicate pixel measurement");
        projection(p); // Validate noise even for observations outside the budget.
        if (!points.count(p.landmark) && seeds.count(p.landmark) &&
            candidates.insert(p.landmark).second) {
            const int col = std::min(7, int(p.pixel.x * 8 / camera.imageSize.width));
            const int row = std::min(5, int(p.pixel.y * 6 / camera.imageSize.height));
            cells[row * 8 + col].push_back(p.landmark);
        }
    }
    for (size_t round = 0; points.size() < size_t(options.landmarks); ++round) {
        bool any = false;
        for (const auto& cell : cells)
            if (round < cell.size() && points.size() < size_t(options.landmarks)) {
                any = true;
                const auto id = cell[round];
                const auto p = seeds.at(id);
                // Initial 3D geometry is an optimizer guess, never a second factor.
                points.emplace(id, Point{{p.x, p.y, p.z}, {}});
            }
        if (!any)
            break;
    }
    for (const auto& p : batch.pixels)
        if (auto it = points.find(p.landmark); it != points.end()) {
            auto& observations = it->second.observations;
            const auto duplicate =
                std::find_if(observations.begin(), observations.end(), [&](const Observation& o) {
                    return o.frame == batch.frame && o.pixel.camera == p.camera;
                });
            if (duplicate == observations.end())
                observations.push_back({batch.frame, p});
        }
    covariance.reset();
}
bool WindowEstimator::Impl::optimize(int iterationBudget) {
    for (auto& e : edges) {
        const auto& x = nodes.at(e.first).x;
        if ((ba::vector(x.data(), 10) - e.integral.gyroBias).norm() > .005 ||
            (ba::vector(x.data(), 13) - e.integral.accelBias).norm() > .05)
            e.integral =
                ba::ImuPreintegration(e.integral.samples, imu, ba::vector(x.data(), 10),
                                      ba::vector(x.data(), 13), e.integral.modelUncertainty);
    }
    ceres::Problem problem;
    for (auto& entry : nodes)
        problem.AddParameterBlock(entry.second.x.data(), 16, new ba::StateManifold);
    problem.AddParameterBlock(gravity.data(), 3, new ba::GravityManifold);
    if (prior.A.rows()) {
        std::vector<double*> parameters;
        for (int frame : prior.frames)
            parameters.push_back(nodes.at(frame).x.data());
        parameters.push_back(gravity.data());
        problem.AddResidualBlock(
            new ba::PriorCost(prior.A, prior.b, prior.anchors, prior.gravityAnchor), nullptr,
            parameters);
    }
    for (const auto& e : edges)
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<ba::ImuFactor, 15, 16, 16, 3>(
                                     new ba::ImuFactor{e.integral}),
                                 nullptr, nodes.at(e.first).x.data(), nodes.at(e.second).x.data(),
                                 gravity.data());
    for (auto& entry : nodes)
        if (entry.second.stationary)
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ba::VelocityFactor, 3, 16>(new ba::VelocityFactor),
                nullptr, entry.second.x.data());
    auto* loss = new ceres::HuberLoss(2.5);
    bool usedLoss = false;
    ceres::ParameterBlockOrdering ordering;
    for (auto& entry : points) {
        auto& point = entry.second;
        if (point.observations.empty())
            continue;
        ordering.AddElementToGroup(point.x.data(), 0);
        for (const auto& o : point.observations) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<ba::ProjectionFactor, 2, 16, 3>(
                    new ba::ProjectionFactor(projection(o.pixel))),
                loss, nodes.at(o.frame).x.data(), point.x.data());
            usedLoss = true;
        }
    }
    if (!usedLoss)
        delete loss;
    for (auto& entry : nodes)
        ordering.AddElementToGroup(entry.second.x.data(), 1);
    ordering.AddElementToGroup(gravity.data(), 1);
    ceres::Solver::Options settings;
    settings.linear_solver_type = usedLoss ? ceres::DENSE_SCHUR : ceres::DENSE_QR;
    settings.linear_solver_ordering = std::make_shared<ceres::ParameterBlockOrdering>(ordering);
    settings.max_num_iterations = iterationBudget > 0 ? iterationBudget : options.iterations;
    settings.num_threads = 1;
    settings.function_tolerance = 1e-5;
    settings.parameter_tolerance = 1e-7;
    settings.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(settings, &problem, &summary);
    diagnostic.initialCost = summary.initial_cost;
    diagnostic.finalCost = summary.final_cost;
    diagnostic.iterations = summary.iterations.size();
    diagnostic.solverMessage = summary.message;
    covariance.reset();
    if (!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost) ||
        summary.final_cost > summary.initial_cost + 1e-6)
        return false;
    for (const auto& entry : nodes)
        if (!finite(entry.second.x))
            return false;
    if (!gravity.allFinite() || std::abs(gravity.norm() - ba::gravityMagnitude) > 1e-6)
        return false;
    return true;
}
WindowEstimator::WindowEstimator(StereoCalibration c, ImuCalibration i, WindowOptions o)
    : impl_(std::make_unique<Impl>(std::move(c), i, o)) {}
WindowEstimator::~WindowEstimator() = default;
WindowEstimator::WindowEstimator(const WindowEstimator& other)
    : impl_(std::make_unique<Impl>(*other.impl_)) {}
WindowEstimator& WindowEstimator::operator=(const WindowEstimator& other) {
    if (this != &other)
        impl_ = std::make_unique<Impl>(*other.impl_);
    return *this;
}
bool WindowEstimator::initialized() const {
    return !impl_->nodes.empty();
}
void WindowEstimator::initialize(const InertialState& s, TimestampNs timestamp, int frame) {
    if (initialized() || !s.initialized || timestamp < 0 || frame < 0 ||
        !std::isfinite(cv::norm(s.gravity)) || std::abs(cv::norm(s.gravity) - 9.81) > .1)
        throw std::invalid_argument("Invalid window initialization");
    auto& w = *impl_;
    const auto x = stateVector(s, w.bodyFromCamera);
    if (!finite(x))
        throw std::invalid_argument("Non-finite initial navigation state");
    w.gravity = eigen(s.gravity).normalized() * ba::gravityMagnitude;
    w.nodes.emplace(frame, Impl::Node{frame, timestamp, x});
    // Finite initialization uncertainty, not hard fixing all 6 pose components.
    // The initial reference anchors the world-frame gauge with finite uncertainty.
    Eigen::Matrix<double, 17, 1> sigma;
    sigma << .05, .05, .05, .01, .01, .01, .2, .2, .2, .03, .03, .03, .5, .5, .5,
        .2 / ba::gravityMagnitude, .2 / ba::gravityMagnitude;
    w.prior = {
        {frame}, {x}, sigma.cwiseInverse().asDiagonal(), Eigen::VectorXd::Zero(17), w.gravity};
}
void WindowEstimator::propagate(const std::vector<ImuSample>& interval) {
    auto& w = *impl_;
    if (!initialized())
        throw std::logic_error("Uninitialized window propagation");
    const auto before = w.current();
    if (interval.size() < 2 || interval.front().timestampNs != before.timestamp)
        throw std::invalid_argument("Window propagation requires a contiguous IMU interval");
    ba::ImuPreintegration p(interval, w.imu, ba::vector(before.x.data(), 10),
                            ba::vector(before.x.data(), 13), true);
    w.diagnostic.imuGyroNoise = p.gyroNoiseDensity.maxCoeff();
    w.diagnostic.imuAccelNoise = p.accelNoiseDensity.maxCoeff();
    auto x = before.x;
    const auto q = ba::quaternion(before.x.data());
    Eigen::Map<Eigen::Quaterniond> nextQ(x.data());
    nextQ = (q * p.rotation).normalized();
    Eigen::Map<V3>(x.data() + 4) = ba::vector(before.x.data(), 4) +
                                   ba::vector(before.x.data(), 7) * p.dt +
                                   w.gravity * (.5 * p.dt * p.dt) + q * p.position;
    Eigen::Map<V3>(x.data() + 7) =
        ba::vector(before.x.data(), 7) + w.gravity * p.dt + q * p.velocity;
    const int frame = before.frame + 1;
    w.nodes.emplace(frame, Impl::Node{frame, interval.back().timestampNs, x});
    w.edges.push_back({before.frame, frame, std::move(p)});
    w.committedFeedback = {};
    w.diagnostic.posteriorOutliers = 0;
    w.diagnostic.removedObservations = 0;
    w.diagnostic.cleanupAttempted = false;
    w.diagnostic.cleanupIterations = 0;
    w.diagnostic.cleanupInitialCost = w.diagnostic.cleanupFinalCost = 0;
    w.diagnostic.accepted = false;
    w.diagnostic.informativeLandmarks = 0;
    w.diagnostic.rejection = 1;
    w.diagnostic.stationaryAttempted = false;
    w.diagnostic.stationaryPosteriorSpeed = -1;
    w.diagnostic.stationaryInnovation = -1;
    w.diagnostic.solverMessage.clear();
    w.diagnostic.inliers = 0;
    w.diagnostic.iterations = 0;
    w.diagnostic.initialCost = 0;
    w.diagnostic.finalCost = 0;
    w.covariance.reset();
    w.trim();
}
bool WindowEstimator::correct(const VisualMeasurements& batch) {
    if (!initialized())
        throw std::logic_error("Uninitialized window correction");
    auto& w = *impl_;
    w.committedFeedback = {};
    w.diagnostic.posteriorOutliers = 0;
    w.diagnostic.removedObservations = 0;
    w.diagnostic.cleanupAttempted = false;
    w.diagnostic.cleanupIterations = 0;
    w.diagnostic.cleanupInitialCost = w.diagnostic.cleanupFinalCost = 0;
    auto backup = w;
    std::set<LandmarkId> supported;
    for (const auto& pixel : batch.pixels)
        if (auto it = w.points.find(pixel.landmark); it != w.points.end())
            for (const auto& old : it->second.observations)
                if (old.frame != batch.frame) {
                    supported.insert(pixel.landmark);
                    break;
                }
    w.add(batch);
    const bool enoughHistory = supported.size() >= size_t(w.options.minInliers);
    const bool solved = enoughHistory && w.optimize();
    bool accepted = solved;
    std::set<LandmarkId> inliers;
    std::set<int> cells;
    auto validGeometry = [&]() {
        inliers.clear();
        cells.clear();
        for (const auto& entry : w.points)
            if (supported.count(entry.first))
                for (const auto& o : entry.second.observations)
                    if (o.frame == batch.frame) {
                        const auto f = w.projection(o.pixel);
                        const V3 p =
                            f.R_camera_imu * (ba::quaternion(w.current().x.data()).conjugate() *
                                              (Eigen::Map<const V3>(entry.second.x.data()) -
                                               ba::vector(w.current().x.data(), 4))) +
                            f.t_camera_imu;
                        if (p.z() <= .02)
                            continue;
                        const Eigen::Vector2d r(f.fx * p.x() / p.z() + f.cx - o.pixel.pixel.x,
                                                f.fy * p.y() / p.z() + f.cy - o.pixel.pixel.y);
                        if (r.norm() < 3.) {
                            inliers.insert(entry.first);
                            const int x =
                                std::min(7, int(o.pixel.pixel.x * 8 / w.camera.imageSize.width));
                            const int y =
                                std::min(5, int(o.pixel.pixel.y * 6 / w.camera.imageSize.height));
                            cells.insert(y * 8 + x);
                        }
                    }
        return inliers.size() >= size_t(w.options.minInliers) &&
               inliers.size() >= .4 * supported.size() && cells.size() >= 4;
    };
    accepted = solved && validGeometry();
    int cleanupRejection = 0;
    if (accepted && w.options.observationFeedback) {
        // Resolve mode uses a provisional classification for deletion. History
        // is advanced once, using the final accepted posterior, below.
        auto feedback = w.classifyObservations(!w.options.resolveAfterPruning);
        w.prunePersistentOutliers(feedback);
        if (w.options.resolveAfterPruning && w.diagnostic.removedObservations) {
            const auto primary = w.diagnostic;
            const bool cleanupSolved = w.optimize(2); // One fixed, finite extra solve.
            const auto cleanup = w.diagnostic;
            w.diagnostic = primary;
            w.diagnostic.cleanupAttempted = true;
            w.diagnostic.cleanupIterations =
                std::max(0, cleanup.iterations - 1); // Exclude Ceres iteration zero.
            w.diagnostic.iterations += w.diagnostic.cleanupIterations;
            w.diagnostic.cleanupInitialCost = cleanup.initialCost;
            w.diagnostic.cleanupFinalCost = cleanup.finalCost;
            w.diagnostic.solverMessage += "; cleanup: " + cleanup.solverMessage;
            // Costs belong to different factor sets. Keep primary and cleanup
            // costs separate instead of presenting their difference as gain.
            accepted = cleanupSolved && validGeometry();
            if (!accepted)
                cleanupRejection = cleanupSolved ? 6 : 5;
        }
        if (accepted && w.options.resolveAfterPruning) {
            auto finalFeedback = w.classifyObservations(true);
            for (const auto& o : feedback.observations)
                if (o.state == ObservationState::Removed)
                    finalFeedback.observations.push_back(o); // Residual at the deletion decision.
            feedback = std::move(finalFeedback);
            w.diagnostic.posteriorOutliers =
                std::count_if(feedback.observations.begin(), feedback.observations.end(),
                              [](const auto& o) { return o.state != ObservationState::Inlier; });
        }
        if (accepted)
            w.committedFeedback = std::move(feedback);
    }
    const auto attempted = w.diagnostic;
    if (!accepted)
        w = std::move(backup);
    w.diagnostic.initialCost = attempted.initialCost;
    w.diagnostic.finalCost = attempted.finalCost;
    w.diagnostic.iterations = attempted.iterations;
    w.diagnostic.solverMessage = attempted.solverMessage;
    w.diagnostic.accepted = accepted;
    w.diagnostic.inliers = inliers.size();
    w.diagnostic.informativeLandmarks = supported.size();
    w.diagnostic.cleanupAttempted = attempted.cleanupAttempted;
    w.diagnostic.cleanupIterations = attempted.cleanupIterations;
    w.diagnostic.cleanupInitialCost = attempted.cleanupInitialCost;
    w.diagnostic.cleanupFinalCost = attempted.cleanupFinalCost;
    w.diagnostic.removedObservations = accepted ? attempted.removedObservations : 0;
    w.diagnostic.rejection = accepted           ? 0
                             : cleanupRejection ? cleanupRejection
                             : !enoughHistory   ? 2
                             : !solved          ? 3
                                                : 4;
    return accepted;
}
void WindowEstimator::seed(const VisualMeasurements& batch) {
    if (!initialized())
        throw std::logic_error("Uninitialized landmark seeding");
    auto& w = *impl_;
    // A committed keyframe replaces its active ID set. Marginalize displaced
    // tracks before admitting new IDs, so a full stale budget cannot block
    // every temporal constraint. Validate on a candidate before changing w.
    auto next = w;
    next.add(batch);
    std::set<LandmarkId> active;
    for (const auto& pixel : batch.pixels)
        active.insert(pixel.landmark);
    std::set<LandmarkId> retired;
    for (const auto& point : w.points)
        if (!active.count(point.first))
            retired.insert(point.first);
    if (!retired.empty())
        w.marginalize(-1, retired);
    w.add(batch);
}
bool WindowEstimator::correctStationary() {
    auto& w = *impl_;
    if (!initialized() || w.current().stationary)
        return false;
    w.diagnostic.stationaryAttempted = true;
    try {
        const auto& P = w.jointCovariance();
        const int offset = 15 * (w.nodes.size() - 1) + 6;
        w.diagnostic.stationaryInnovation =
            ba::zeroVelocityNis(ba::vector(w.current().x.data(), 7), P.block<3, 3>(offset, offset));
    } catch (const std::runtime_error&) {
        w.diagnostic.solverMessage = "Zero-velocity covariance is unavailable";
        return false;
    }
    if (!std::isfinite(w.diagnostic.stationaryInnovation) ||
        w.diagnostic.stationaryInnovation > ba::zeroVelocityGate) {
        w.diagnostic.solverMessage = "Zero-velocity innovation rejected";
        return false;
    }
    auto backup = w;
    w.current().stationary = true;
    // Only zero velocity is inserted. The same IMU sample means are not reused
    // as independent bias/gravity observations on top of preintegration.
    const bool solved = w.optimize();
    w.diagnostic.stationaryAttempted = true;
    w.diagnostic.stationaryPosteriorSpeed = ba::vector(w.current().x.data(), 7).norm();
    // Acceptance is based on the prior innovation and a usable MAP solution.
    // A finite-noise measurement need not force its posterior velocity below
    // an unrelated fixed threshold. Rejected solves remain transactional.
    if (solved)
        return true;
    const auto attempted = w.diagnostic;
    w = std::move(backup);
    w.diagnostic = attempted;
    return false;
}
void WindowEstimator::rememberReference(int frame) {
    auto& w = *impl_;
    if (!w.nodes.count(frame))
        throw std::logic_error("Reference is outside the optimization window");
    if (std::find(w.references.begin(), w.references.end(), frame) != w.references.end())
        throw std::logic_error("Duplicate window reference");
    w.references.push_back(frame);
    if (w.references.size() > 6)
        w.references.erase(w.references.begin() +
                           (w.references.front() == w.protectedReference ? 1 : 0));
}
void WindowEstimator::protectReference(int frame) {
    if (std::find(impl_->references.begin(), impl_->references.end(), frame) ==
        impl_->references.end())
        throw std::logic_error("Protected reference is not retained");
    impl_->protectedReference = frame;
}
std::optional<cv::Matx44d> WindowEstimator::referencePose(int frame) const {
    if (auto it = impl_->nodes.find(frame); it != impl_->nodes.end())
        return impl_->pose(it->second.x);
    return std::nullopt;
}
std::optional<cv::Point3d> WindowEstimator::landmarkPosition(LandmarkId id) const {
    if (auto it = impl_->points.find(id); it != impl_->points.end()) {
        const auto& p = it->second.x;
        return cv::Point3d(p[0], p[1], p[2]);
    }
    return std::nullopt;
}
std::vector<LandmarkSeed> WindowEstimator::optimizedLandmarks() const {
    std::vector<LandmarkSeed> result;
    for (const auto& entry : impl_->points)
        result.push_back({entry.first, {entry.second.x[0], entry.second.x[1], entry.second.x[2]}});
    return result;
}
std::pair<double, double> WindowEstimator::relativeUncertainty(int reference) const {
    const auto& w = *impl_;
    const double infinity = std::numeric_limits<double>::infinity();
    if (!initialized() || !w.nodes.count(reference))
        return {infinity, infinity};
    try {
        const auto order = w.frames();
        const auto& P = w.jointCovariance();
        Eigen::MatrixXd J = Eigen::MatrixXd::Zero(6, P.cols());
        const auto& a = w.nodes.at(reference).x;
        const auto& b = w.current().x;
        const M4 relative = inverse(eigen(w.pose(a))) * eigen(w.pose(b));
        ba::StateManifold manifold;
        for (size_t k = 0; k < order.size(); ++k)
            if (order[k] == reference || order[k] == w.current().frame)
                for (int j = 0; j < 15; ++j) {
                    double delta[15] = {};
                    delta[j] = 1e-6;
                    ba::State perturbed;
                    manifold.Plus(w.nodes.at(order[k]).x.data(), delta, perturbed.data());
                    const auto ar = order[k] == reference ? perturbed : a,
                               br = order[k] == w.current().frame ? perturbed : b;
                    const M4 transformed = inverse(eigen(w.pose(ar))) * eigen(w.pose(br));
                    J.block<3, 1>(0, 15 * k + j) =
                        (transformed.topRightCorner<3, 1>() - relative.topRightCorner<3, 1>()) /
                        1e-6;
                    J.block<3, 1>(3, 15 * k + j) =
                        inertial::log(relative.topLeftCorner<3, 3>().transpose() *
                                      transformed.topLeftCorner<3, 3>()) /
                        1e-6;
                }
        const Eigen::Matrix<double, 6, 6> C = J * P * J.transpose();
        return {std::sqrt(std::max(0., C.topLeftCorner<3, 3>().trace())),
                std::sqrt(std::max(0., C.bottomRightCorner<3, 3>().trace()))};
    } catch (const std::runtime_error&) {
        return {infinity, infinity};
    }
}
InertialState WindowEstimator::state() const {
    InertialState s;
    if (!initialized())
        return s;
    const auto& w = *impl_;
    const auto& x = w.current().x;
    s.initialized = true;
    s.T_world_rect = w.pose(x);
    s.gravity = cvvec(w.gravity);
    s.velocity = cvvec(ba::vector(x.data(), 7));
    s.gyroBias = cvvec(ba::vector(x.data(), 10));
    s.accelBias = cvvec(ba::vector(x.data(), 13));
    s.positionStd = -1; // Unavailable until covariance is requested for recovery.
    if (w.covariance) {
        const int offset = 15 * (w.nodes.size() - 1) + 3;
        s.positionStd = std::sqrt(std::max(0., w.covariance->block<3, 3>(offset, offset).trace()));
    }
    s.innovation = w.diagnostic.finalCost;
    return s;
}
EstimationFeedback WindowEstimator::feedback() const {
    return impl_->committedFeedback;
}
WindowDiagnostics WindowEstimator::diagnostics() const {
    auto d = impl_->diagnostic;
    d.states = impl_->nodes.size();
    d.landmarks = impl_->points.size();
    d.observations = 0;
    for (const auto& entry : impl_->points)
        d.observations += entry.second.observations.size();
    if (initialized() && impl_->options.auditEvery > 0 &&
        impl_->current().frame % impl_->options.auditEvery == 0)
        d.audit = impl_->audit();
    return d;
}
}
