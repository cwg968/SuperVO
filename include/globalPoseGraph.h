#pragma once
#include <sensorTypes.h>
#include <memory>
#include <vector>
namespace superVO {
struct GraphEdge {
    int first = 0, second = 0;
    cv::Matx44d firstFromSecond = cv::Matx44d::eye();
    bool loop = false;
    double weight = 1;
};
class GlobalPoseGraph {
  public:
    explicit GlobalPoseGraph(cv::Vec3d gravity);
    ~GlobalPoseGraph();
    void addNode(const cv::Matx44d& localPose);
    // Rejected solves never modify the published graph or retained edges.
    bool addLoop(int first, int second, const cv::Matx44d& firstFromSecond);
    std::vector<cv::Matx44d> poses() const;
    std::vector<GraphEdge> edges() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
