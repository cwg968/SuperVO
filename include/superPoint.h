#pragma once
#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>

namespace superVO {
cv::Mat prepareSuperPointImage(const cv::Mat& image, cv::Size expected);
std::vector<cv::Point2f> decodeSuperPoint(const cv::Mat& logits, cv::Size imageSize,
                                          float threshold = 0.015f, int maxPoints = 1000,
                                          int radius = 4);
struct SuperPointOutput {
    cv::Mat logits, descriptors;
}; // owned raw NCHW tensors
struct SuperPointFeatures {
    std::vector<cv::Point2f> pixels;
    cv::Mat descriptors; // Nx256 CV_32F, one unit descriptor per pixel
};
cv::Mat sampleSuperPointDescriptors(const cv::Mat& coarse, cv::Size inputSize,
                                    const std::vector<cv::Point2f>& pixels);
class SuperPointDetector {
  public:
    SuperPointDetector(const std::string& weights, const std::string& backend = "cpu",
                       const std::string& engine = "", cv::Size size = {});
    ~SuperPointDetector();
    // Owned NCHW detector logits, also exposed for CPU/GPU numerical validation.
    SuperPointOutput infer(const cv::Mat& image);
    SuperPointFeatures extract(const cv::Mat& image, float threshold = 0.015f,
                               int maxPoints = 1000);
    cv::Mat inferLogits(const cv::Mat& image);
    std::vector<cv::Point2f> detect(const cv::Mat& gray, float threshold = 0.015f,
                                    int maxPoints = 1000);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
