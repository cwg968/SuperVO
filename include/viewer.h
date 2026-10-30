#pragma once
#include <visualizationSnapshot.h>
#include <memory>
#include <optional>
#include <string>

namespace superVO {
struct ViewerOptions {
    bool enabled = false, waitAtEnd = true;
};
struct ViewerFrame {
    int frameIndex = 0;
    TimestampNs timestampNs = 0;
    cv::Mat leftImage;
    std::optional<cv::Matx44d> T_world_rect;
    VisualizationSnapshot geometry;
    std::string status;
    int inliers = 0;
    double processingMs = 0;
    bool keyframe = false;
};
// The render thread owns all Pangolin/OpenGL resources. A single latest-frame
// slot bounds display latency; rendering never touches or updates the estimator.
class Viewer {
  public:
    Viewer(const cv::Mat& intrinsics, cv::Size imageSize);
    ~Viewer();
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;
    void submit(ViewerFrame frame);
    bool waitUntilRunning();
    void finish(bool waitForClose);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
