#include "windowInternal.h"
#include <stdexcept>

namespace superVO {
namespace {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
struct Stack {
    std::vector<ba::LinearSystem> chunks;
    int rows = 0, columns;
    explicit Stack(int n) : columns(n) {}
    void append(ba::LinearSystem block) {
        rows += block.A.rows();
        chunks.push_back(std::move(block));
    }
    ba::LinearSystem finish() const {
        ba::LinearSystem s{Matrix::Zero(rows, columns), Vector::Zero(rows)};
        int row = 0;
        for (const auto& c : chunks) {
            s.A.middleRows(row, c.A.rows()) = c.A;
            s.b.segment(row, c.b.size()) = c.b;
            row += c.A.rows();
        }
        return s;
    }
};
}
std::vector<int> WindowEstimator::Impl::frames() const {
    std::vector<int> f;
    for (const auto& entry : nodes)
        f.push_back(entry.first);
    return f;
}
ba::LinearSystem WindowEstimator::Impl::linearize(int removed, const std::set<LandmarkId>& retired,
                                                  bool all) const {
    const auto order = frames();
    const int n = 15 * order.size() + 2;
    std::map<int, int> offset;
    std::map<int, Eigen::Matrix<double, 16, 15, Eigen::RowMajor>> plus;
    ba::StateManifold manifold;
    ba::GravityManifold gravityManifold;
    Eigen::Matrix<double, 3, 2, Eigen::RowMajor> gravityPlus;
    gravityManifold.PlusJacobian(gravity.data(), gravityPlus.data());
    for (size_t k = 0; k < order.size(); ++k) {
        offset[order[k]] = 15 * k;
        manifold.PlusJacobian(nodes.at(order[k]).x.data(), plus[order[k]].data());
    }
    Stack stack(n);
    if (prior.A.rows()) {
        ba::PriorCost cost(prior.A, prior.b, prior.anchors, prior.gravityAnchor);
        std::vector<const double*> inputs;
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, 16, Eigen::RowMajor>> jac;
        jac.reserve(prior.frames.size());
        for (int f : prior.frames) {
            inputs.push_back(nodes.at(f).x.data());
            jac.emplace_back(cost.num_residuals(), 16);
        }
        inputs.push_back(gravity.data());
        Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> gravityJacobian(
            cost.num_residuals(), 3);
        std::vector<double*> output;
        for (auto& J : jac)
            output.push_back(J.data());
        output.push_back(gravityJacobian.data());
        ba::LinearSystem s{Matrix::Zero(cost.num_residuals(), n), Vector(cost.num_residuals())};
        cost.Evaluate(inputs.data(), s.b.data(), output.data());
        for (size_t k = 0; k < prior.frames.size(); ++k)
            s.A.middleCols(offset.at(prior.frames[k]), 15) = jac[k] * plus.at(prior.frames[k]);
        s.A.rightCols(2) = gravityJacobian * gravityPlus;
        stack.append(std::move(s));
    }
    for (const auto& edge : edges)
        if (all || edge.first == removed || edge.second == removed) {
            ceres::AutoDiffCostFunction<ba::ImuFactor, 15, 16, 16, 3> cost(
                new ba::ImuFactor{edge.integral});
            const double* inputs[] = {nodes.at(edge.first).x.data(), nodes.at(edge.second).x.data(),
                                      gravity.data()};
            Eigen::Matrix<double, 15, 16, Eigen::RowMajor> a, b;
            Eigen::Matrix<double, 15, 3, Eigen::RowMajor> g;
            double* output[] = {a.data(), b.data(), g.data()};
            ba::LinearSystem s{Matrix::Zero(15, n), Vector(15)};
            cost.Evaluate(inputs, s.b.data(), output);
            s.A.middleCols(offset.at(edge.first), 15) = a * plus.at(edge.first);
            s.A.middleCols(offset.at(edge.second), 15) = b * plus.at(edge.second);
            s.A.rightCols(2) = g * gravityPlus;
            stack.append(std::move(s));
        }
    for (const auto& entry : nodes)
        if (entry.second.stationary && (all || entry.first == removed)) {
            ceres::AutoDiffCostFunction<ba::VelocityFactor, 3, 16> cost(new ba::VelocityFactor);
            const double* inputs[] = {entry.second.x.data()};
            Eigen::Matrix<double, 3, 16, Eigen::RowMajor> J;
            double* output[] = {J.data()};
            ba::LinearSystem s{Matrix::Zero(3, n), Vector(3)};
            cost.Evaluate(inputs, s.b.data(), output);
            s.A.middleCols(offset.at(entry.first), 15) = J * plus.at(entry.first);
            stack.append(std::move(s));
        }
    for (const auto& entry : points)
        if (all || retired.count(entry.first)) {
            const auto& point = entry.second;
            ba::LinearSystem s{Matrix::Zero(2 * point.observations.size(), 3 + n),
                               Vector(2 * point.observations.size())};
            int row = 0;
            for (const auto& observation : point.observations) {
                ceres::AutoDiffCostFunction<ba::ProjectionFactor, 2, 16, 3> cost(
                    new ba::ProjectionFactor(projection(observation.pixel)));
                const double* inputs[] = {nodes.at(observation.frame).x.data(), point.x.data()};
                Eigen::Matrix<double, 2, 16, Eigen::RowMajor> state;
                Eigen::Matrix<double, 2, 3, Eigen::RowMajor> landmark;
                double* output[] = {state.data(), landmark.data()};
                Eigen::Vector2d r;
                if (!cost.Evaluate(inputs, r.data(), output))
                    continue;
                const double norm = r.norm(), weight = norm <= 2.5 ? 1. : std::sqrt(2.5 / norm);
                s.b.segment<2>(row) = weight * r;
                s.A.block(row, 0, 2, 3) = weight * landmark;
                s.A.block(row, 3 + offset.at(observation.frame), 2, 15) =
                    weight * state * plus.at(observation.frame);
                row += 2;
            }
            s.A.conservativeResize(row, Eigen::NoChange);
            s.b.conservativeResize(row);
            if (row)
                stack.append(ba::eliminate(s.A, s.b, 3));
        }
    return stack.finish();
}
void WindowEstimator::Impl::marginalize(int removed, const std::set<LandmarkId>& retired) {
    auto s = linearize(removed, retired, false);
    auto order = frames();
    if (removed >= 0) {
        const auto it = std::find(order.begin(), order.end(), removed);
        if (it == order.end())
            throw std::logic_error("Unknown marginalization state");
        const int index = std::distance(order.begin(), it);
        Matrix reordered(s.A.rows(), s.A.cols());
        reordered.leftCols(15) = s.A.middleCols(index * 15, 15);
        int col = 15;
        for (size_t k = 0; k < order.size(); ++k)
            if (int(k) != index) {
                reordered.middleCols(col, 15) = s.A.middleCols(k * 15, 15);
                col += 15;
            }
        reordered.rightCols(2) = s.A.rightCols(2);
        s = ba::eliminate(reordered, s.b, 15);
        order.erase(it);
    }
    auto compressed = ba::compress(s.A, s.b);
    prior = {};
    prior.frames = order;
    prior.A = std::move(compressed.A);
    prior.b = std::move(compressed.b);
    prior.gravityAnchor = gravity;
    for (int frame : order)
        prior.anchors.push_back(nodes.at(frame).x);
    for (auto id : retired)
        points.erase(id);
    if (removed >= 0) {
        nodes.erase(removed);
        edges.erase(std::remove_if(
                        edges.begin(), edges.end(),
                        [&](const Edge& e) { return e.first == removed || e.second == removed; }),
                    edges.end());
        // Observations in a redundant non-host frame are discarded. Keeping
        // them in a prior would couple retained landmark blocks and invalidate
        // their independent Schur elimination. Host-track factors are instead
        // marginalized in full above. No observation is counted twice.
        for (auto& entry : points) {
            auto& observations = entry.second.observations;
            observations.erase(
                std::remove_if(observations.begin(), observations.end(),
                               [&](const Observation& o) { return o.frame == removed; }),
                observations.end());
        }
        ++diagnostic.marginalizedStates;
    }
    diagnostic.marginalizedLandmarks += retired.size();
    diagnostic.priorRank = prior.A.rows();
    covariance.reset();
}
void WindowEstimator::Impl::trim() {
    while (int(nodes.size()) > options.states) {
        int removed = -1;
        for (const auto& entry : nodes)
            if (entry.first != current().frame &&
                std::find(references.begin(), references.end(), entry.first) == references.end()) {
                removed = entry.first;
                break;
            }
        if (removed < 0)
            throw std::logic_error("Optimization window cannot evict a protected reference");
        std::set<LandmarkId> retired;
        for (const auto& entry : points) {
            const auto& observations = entry.second.observations;
            if (observations.front().frame == removed ||
                current().timestamp - nodes.at(observations.back().frame).timestamp > 500000000)
                retired.insert(entry.first);
        }
        marginalize(removed, retired);
    }
}
const Eigen::MatrixXd& WindowEstimator::Impl::jointCovariance() const {
    if (!covariance) {
        const auto s = linearize(-1, {}, true);
        const Matrix H = s.A.transpose() * s.A;
        Eigen::LDLT<Matrix> factor(H);
        if (factor.info() != Eigen::Success || !factor.isPositive())
            throw std::runtime_error("Unobservable/indefinite window covariance");
        covariance = factor.solve(Matrix::Identity(H.rows(), H.cols()));
        if (!covariance->allFinite())
            throw std::runtime_error("Non-finite window covariance");
    }
    return *covariance;
}
}
