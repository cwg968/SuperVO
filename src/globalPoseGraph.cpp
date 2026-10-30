#include <globalPoseGraph.h>
#include <ceres/ceres.h>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <stdexcept>
namespace superVO {
namespace {
Eigen::Matrix4d eigen(const cv::Matx44d& t) {
    Eigen::Matrix4d out;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out(i, j) = t(i, j);
    return out;
}
cv::Matx44d opencv(const Eigen::Matrix4d& t) {
    cv::Matx44d out;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            out(i, j) = t(i, j);
    return out;
}
struct EdgeCost {
    Eigen::Matrix3d ri, rj, measured;
    Eigen::Vector3d translation;
    double positionSigma, angleSigma;
    template <typename T> bool operator()(const T* a, const T* b, const T* s, T* r) const {
        using std::sin;
        using std::cos;
        using std::atan2;
        auto rz = [](T v) {
            Eigen::Matrix<T, 3, 3> m;
            m << cos(v), -sin(v), T(0), sin(v), cos(v), T(0), T(0), T(0), T(1);
            return m;
        };
        Eigen::Matrix<T, 3, 3> ra = rz(a[0]) * ri.cast<T>(), rb = rz(b[0]) * rj.cast<T>();
        Eigen::Matrix<T, 3, 1> d;
        d << b[1] - a[1], b[2] - a[2], b[3] - a[3];
        Eigen::Matrix<T, 3, 1> e = ra.transpose() * d - translation.cast<T>();
        for (int i = 0; i < 3; ++i)
            r[i] = s[0] * e[i] / T(positionSigma);
        Eigen::Matrix<T, 3, 3> dr = measured.cast<T>().transpose() * ra.transpose() * rb;
        // Signed rotation around gravity expressed in camera i coordinates.
        Eigen::Matrix<T, 3, 1> axis = rj.transpose().col(2).cast<T>();
        Eigen::Matrix<T, 3, 1> skew;
        skew << dr(2, 1) - dr(1, 2), dr(0, 2) - dr(2, 0), dr(1, 0) - dr(0, 1);
        T theta = atan2(axis.dot(skew), dr.trace() - T(1));
        r[3] = s[0] * theta / T(angleSigma);
        return true;
    }
};
struct SwitchPrior {
    template <typename T> bool operator()(const T* s, T* r) const {
        r[0] = T(3) * (T(1) - s[0]);
        return true;
    }
};
}
struct GlobalPoseGraph::Impl {
    Eigen::Matrix3d alignment;
    std::vector<Eigen::Matrix4d> local;
    std::vector<std::array<double, 4>> states;
    std::vector<GraphEdge> constraints;
    explicit Impl(cv::Vec3d g) {
        Eigen::Vector3d gravity(g[0], g[1], g[2]);
        if (!gravity.allFinite() || gravity.norm() < 1)
            throw std::invalid_argument("Invalid graph gravity");
        alignment =
            Eigen::Quaterniond::FromTwoVectors(-gravity.normalized(), Eigen::Vector3d::UnitZ())
                .toRotationMatrix();
    }
    Eigen::Matrix4d pose(size_t i) const {
        Eigen::Matrix4d p = Eigen::Matrix4d::Identity();
        p.topLeftCorner<3, 3>() =
            Eigen::AngleAxisd(states[i][0], Eigen::Vector3d::UnitZ()).toRotationMatrix() *
            local[i].topLeftCorner<3, 3>();
        for (int j = 0; j < 3; ++j)
            p(j, 3) = states[i][j + 1];
        Eigen::Matrix4d inv = Eigen::Matrix4d::Identity();
        inv.topLeftCorner<3, 3>() = alignment.transpose();
        return inv * p;
    }
};
GlobalPoseGraph::GlobalPoseGraph(cv::Vec3d g) : impl_(std::make_unique<Impl>(g)) {}
GlobalPoseGraph::~GlobalPoseGraph() = default;
void GlobalPoseGraph::addNode(const cv::Matx44d& pose) {
    if (!cv::checkRange(cv::Mat(pose)))
        throw std::invalid_argument("Invalid graph pose");
    auto& w = *impl_;
    Eigen::Matrix4d a = Eigen::Matrix4d::Identity();
    a.topLeftCorner<3, 3>() = w.alignment;
    Eigen::Matrix4d p = a * eigen(pose);
    size_t n = w.local.size();
    std::array<double, 4> initial{0, p(0, 3), p(1, 3), p(2, 3)};
    if (n) {
        initial[0] = w.states.back()[0];
        Eigen::Matrix3d r =
            Eigen::AngleAxisd(initial[0], Eigen::Vector3d::UnitZ()).toRotationMatrix();
        Eigen::Vector3d previous(w.states.back()[1], w.states.back()[2], w.states.back()[3]);
        Eigen::Vector3d position =
            previous + r * (p.topRightCorner<3, 1>() - w.local.back().topRightCorner<3, 1>());
        for (int j = 0; j < 3; ++j)
            initial[j + 1] = position[j];
    }
    w.local.push_back(p);
    w.states.push_back(initial);
    if (n)
        w.constraints.push_back(
            {int(n - 1), int(n), opencv(w.local[n - 1].inverse() * p), false, 1});
}
bool GlobalPoseGraph::addLoop(int first, int second, const cv::Matx44d& measurement) {
    auto& w = *impl_;
    if (first < 0 || second <= first || second >= int(w.states.size()) ||
        !cv::checkRange(cv::Mat(measurement)))
        throw std::invalid_argument("Invalid graph edge");
    auto states = w.states;
    auto edges = w.constraints;
    edges.push_back({first, second, measurement, true, 1});
    std::vector<double> switches(edges.size(), 1);
    ceres::Problem problem;
    for (auto& x : states)
        problem.AddParameterBlock(x.data(), 4);
    problem.SetParameterBlockConstant(states.front().data());
    for (size_t k = 0; k < edges.size(); ++k) {
        const auto& e = edges[k];
        auto z = eigen(e.firstFromSecond);
        auto* cost = new ceres::AutoDiffCostFunction<EdgeCost, 4, 4, 4, 1>(
            new EdgeCost{w.local[e.first].topLeftCorner<3, 3>(),
                         w.local[e.second].topLeftCorner<3, 3>(), z.topLeftCorner<3, 3>(),
                         z.topRightCorner<3, 1>(), e.loop ? .06 : .10, e.loop ? .025 : .04});
        problem.AddResidualBlock(cost, e.loop ? new ceres::HuberLoss(3) : nullptr,
                                 states[e.first].data(), states[e.second].data(), &switches[k]);
        if (e.loop) {
            problem.SetParameterLowerBound(&switches[k], 0, 0);
            problem.SetParameterUpperBound(&switches[k], 0, 1);
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SwitchPrior, 1, 1>(new SwitchPrior), nullptr,
                &switches[k]);
        } else
            problem.SetParameterBlockConstant(&switches[k]);
    }
    ceres::Solver::Options settings;
    settings.max_num_iterations = 40;
    settings.num_threads = 1;
    settings.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    ceres::Solver::Summary summary;
    ceres::Solve(settings, &problem, &summary);
    if (!summary.IsSolutionUsable() || !std::isfinite(summary.final_cost) ||
        summary.final_cost > summary.initial_cost || switches.back() < .5)
        return false;
    for (const auto& s : states)
        for (double v : s)
            if (!std::isfinite(v))
                return false;
    for (size_t i = 0; i < edges.size(); ++i)
        edges[i].weight = switches[i];
    w.states = std::move(states);
    w.constraints = std::move(edges);
    return true;
}
std::vector<cv::Matx44d> GlobalPoseGraph::poses() const {
    std::vector<cv::Matx44d> out;
    for (size_t i = 0; i < impl_->states.size(); ++i)
        out.push_back(opencv(impl_->pose(i)));
    return out;
}
std::vector<GraphEdge> GlobalPoseGraph::edges() const {
    return impl_->constraints;
}
}
