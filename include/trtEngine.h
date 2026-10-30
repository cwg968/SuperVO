#pragma once
#include <superPoint.h>
#include <memory>
#include <string>
#include <vector>

namespace superVO {
// TensorRT 10, fixed NCHW FP32 tensors, one image per inference.
class trtEngine {
  public:
    trtEngine(int height, int width, int maxBatchSize, std::string engineFile);
    ~trtEngine();
    trtEngine(const trtEngine&) = delete;
    trtEngine& operator=(const trtEngine&) = delete;
    void buildTRTEngine(const std::string& weightsPath);
    cv::Mat infer(const cv::Mat& image);
    SuperPointOutput inferOutputs(const cv::Mat& image, bool withDescriptors = true);
    void getPoints(cv::Mat& image, std::vector<cv::Point3f>& points, cv::Mat& descriptors,
                   float threshold = 0.015f, int maxPoints = 1000);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
