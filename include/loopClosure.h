#pragma once
#include <loopGeometry.h>
#include <globalPoseGraph.h>
namespace superVO {
struct LoopTrial {
    int current = 0, reference = 0;
    double similarity = 0, milliseconds = 0;
    LoopVerification geometry;
    // 0 geometry rejected; 1 tentative; 2 temporal rejection; 3 graph rejected; 4 accepted.
    int decision = 0;
};
struct LoopReport {
    std::vector<LoopKeyframe> keyframes;
    std::vector<cv::Matx44d> globalPoses;
    std::vector<LoopTrial> trials;
    std::vector<GraphEdge> edges;
    size_t version = 0, queuePeak = 0;
};
struct LoopCorrection {
    cv::Matx44d mapFromOdom = cv::Matx44d::eye();
    size_t version = 0;
    TimestampNs anchor = 0;
};
// Background worker owns all archives and graph mutation. Only immutable
// keyframe messages enter, and correction snapshots leave the worker.
class LoopClosure {
  public:
    LoopClosure(StereoCalibration, LoopOptions);
    ~LoopClosure();
    void submit(LoopKeyframe);
    LoopCorrection correction() const;
    LoopReport finish();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
