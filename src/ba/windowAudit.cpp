#include "windowInternal.h"
#include <algorithm>
#include <cmath>

namespace superVO {
namespace {
void accumulate(ResidualAudit& summary, const double* r, int n, bool valid, bool robust = false) {
    ++summary.blocks;
    summary.dimensions += n;
    double squared = 0, maximum = 0;
    for (int k = 0; k < n; ++k) {
        valid = valid && std::isfinite(r[k]);
        squared += r[k] * r[k];
        maximum = std::max(maximum, std::abs(r[k]));
    }
    if (!valid || !std::isfinite(squared)) {
        ++summary.invalidBlocks;
        return;
    }
    summary.squaredNorm += squared;
    summary.maxAbs = std::max(summary.maxAbs, maximum);
    // Same 2.5-sigma Huber objective as the optimizer; report rho(s),
    // without the Ceres 1/2 convention, alongside unrobust whitened energy.
    summary.robustSquaredNorm += robust && squared > 6.25 ? 5 * std::sqrt(squared) - 6.25 : squared;
}
}
WindowAudit WindowEstimator::Impl::audit() const {
    WindowAudit snapshot;
    snapshot.frame = current().frame;
    for (const auto& entry : points)
        for (const auto& o : entry.second.observations) {
            double r[2]{};
            const bool valid =
                projection(o.pixel)(nodes.at(o.frame).x.data(), entry.second.x.data(), r);
            accumulate(snapshot.residuals[o.pixel.camera], r, 2, valid, true);
        }
    for (const auto& edge : edges) {
        double r[15]{};
        const bool valid = ba::ImuFactor{edge.integral}(
            nodes.at(edge.first).x.data(), nodes.at(edge.second).x.data(), gravity.data(), r);
        for (int group = 0; group < 5; ++group)
            accumulate(snapshot.residuals[2 + group], r + 3 * group, 3, valid);
        const auto& p = edge.integral;
        ImuWeightAudit weight;
        weight.first = edge.first;
        weight.second = edge.second;
        weight.seconds = p.dt;
        weight.modelUncertainty = p.modelUncertainty;
        for (int k = 0; k < 3; ++k) {
            weight.gyroDensity[k] = p.gyroNoiseDensity[k];
            weight.accelDensity[k] = p.accelNoiseDensity[k];
        }
        for (int k = 0; k < 15; ++k) {
            weight.covarianceDiagonal[k] = p.covariance(k, k);
            weight.whitenerDiagonal[k] = p.sqrtInformation(k, k);
        }
        weight.whiteningError =
            (p.sqrtInformation * p.covariance * p.sqrtInformation.transpose() - ba::M15::Identity())
                .cwiseAbs()
                .maxCoeff();
        snapshot.imu.push_back(weight);
    }
    if (prior.A.rows()) {
        ba::PriorCost cost(prior.A, prior.b, prior.anchors, prior.gravityAnchor);
        std::vector<const double*> parameters;
        for (int frame : prior.frames)
            parameters.push_back(nodes.at(frame).x.data());
        parameters.push_back(gravity.data());
        Eigen::VectorXd r = Eigen::VectorXd::Zero(prior.A.rows());
        const bool valid = cost.Evaluate(parameters.data(), r.data(), nullptr);
        accumulate(snapshot.residuals[7], r.data(), r.size(), valid);
    }
    for (const auto& entry : nodes)
        if (entry.second.stationary) {
            double r[3]{};
            const bool valid = ba::VelocityFactor{}(entry.second.x.data(), r);
            accumulate(snapshot.residuals[8], r, 3, valid);
        }
    return snapshot;
}
}
