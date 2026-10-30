#pragma once
#include <windowEstimator.h>
#include "factors.h"
#include "squareRoot.h"
#include "../inertialMath.h"
#include <map>
#include <deque>
#include <set>

namespace superVO {
struct WindowEstimator::Impl {
    struct Node {
        int frame;
        TimestampNs timestamp;
        ba::State x;
        bool stationary = false;
    };
    struct Observation {
        int frame;
        PixelMeasurement pixel;
        unsigned outlierStreak = 0;
        int lastEvaluatedFrame = -1;
    };
    struct Point {
        std::array<double, 3> x;
        std::vector<Observation> observations;
    };
    struct Edge {
        int first, second;
        ba::ImuPreintegration integral;
    };
    struct Prior {
        std::vector<int> frames;
        std::vector<ba::State> anchors;
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        Eigen::Vector3d gravityAnchor = Eigen::Vector3d::Zero();
    } prior;
    StereoCalibration camera;
    ImuCalibration imu;
    WindowOptions options;
    Eigen::Matrix4d bodyFromCamera;
    Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
    std::map<int, Node> nodes;
    std::map<LandmarkId, Point> points;
    std::vector<Edge> edges;
    std::deque<int> references;
    int protectedReference = -1;
    WindowDiagnostics diagnostic;
    EstimationFeedback committedFeedback;
    mutable std::optional<Eigen::MatrixXd> covariance;
    Impl(StereoCalibration c, ImuCalibration i, WindowOptions o);
    Node& current() {
        return nodes.rbegin()->second;
    }
    const Node& current() const {
        return nodes.rbegin()->second;
    }
    cv::Matx44d pose(const ba::State& x) const;
    ba::ProjectionFactor projection(const PixelMeasurement& pixel) const;
    void add(const VisualMeasurements& measurements);
    bool optimize(int iterationBudget = 0);
    EstimationFeedback classifyObservations(bool updateHistory);
    void prunePersistentOutliers(EstimationFeedback& feedback);
    void trim();
    void marginalize(int removed, const std::set<LandmarkId>& retired);
    ba::LinearSystem linearize(int removed, const std::set<LandmarkId>& retired, bool all) const;
    const Eigen::MatrixXd& jointCovariance() const;
    std::vector<int> frames() const;
    WindowAudit audit() const;
};
}
