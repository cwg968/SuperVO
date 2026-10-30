#include <superPoint.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <stdexcept>
#ifdef SUPERVO_WITH_TENSORRT
#include <trtEngine.h>
#endif

namespace superVO {
cv::Mat prepareSuperPointImage(const cv::Mat& image, cv::Size expected) {
    if (image.empty() || image.depth() != CV_8U || image.size() != expected)
        throw std::runtime_error(
            "SuperPoint expects a nonempty uint8 image matching the engine dimensions");
    cv::Mat gray;
    if (image.channels() == 1)
        gray = image;
    else if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        throw std::runtime_error("SuperPoint expects grayscale or BGR input");
    cv::Mat normalized;
    gray.convertTo(normalized, CV_32F, 1. / 255.);
    return normalized;
}
std::vector<cv::Point2f> decodeSuperPoint(const cv::Mat& logits, cv::Size size, float threshold,
                                          int maxPoints, int radius) {
    if (logits.dims != 4 || logits.size[0] != 1 || logits.size[1] != 65 ||
        logits.type() != CV_32F || !logits.isContinuous() || logits.size[2] * 8 < size.height ||
        logits.size[3] * 8 < size.width || threshold <= 0 || threshold >= 1 || maxPoints <= 0 ||
        radius < 0)
        throw std::runtime_error("Invalid SuperPoint detector output or configuration");
    int h = logits.size[2], w = logits.size[3], plane = h * w;
    const float* data = logits.ptr<float>();
    std::vector<cv::Point3f> candidates;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int offset = y * w + x;
            float maximum = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < 65; ++c)
                maximum = std::max(maximum, data[c * plane + offset]);
            if (!std::isfinite(maximum))
                throw std::runtime_error("Non-finite SuperPoint logits");
            double sum = 0;
            float probability[65];
            for (int c = 0; c < 65; ++c) {
                float logit = data[c * plane + offset];
                if (!std::isfinite(logit))
                    throw std::runtime_error("Non-finite SuperPoint logits");
                probability[c] = std::exp(logit - maximum);
                sum += probability[c];
            }
            for (int c = 0; c < 64; ++c) {
                int px = x * 8 + c % 8, py = y * 8 + c / 8;
                float score = probability[c] / sum;
                if (score >= threshold && px >= 4 && py >= 4 && px < size.width - 4 &&
                    py < size.height - 4)
                    candidates.emplace_back(px, py, score);
            }
        }
    std::stable_sort(candidates.begin(), candidates.end(),
                     [](const cv::Point3f& a, const cv::Point3f& b) { return a.z > b.z; });
    cv::Mat blocked = cv::Mat::zeros(size, CV_8U);
    std::vector<cv::Point2f> points;
    for (const auto& p : candidates) {
        if (blocked.at<uchar>(int(p.y), int(p.x)))
            continue;
        points.emplace_back(p.x, p.y);
        cv::Rect region(
            std::max(0, int(p.x) - radius), std::max(0, int(p.y) - radius),
            std::min(size.width - 1, int(p.x) + radius) - std::max(0, int(p.x) - radius) + 1,
            std::min(size.height - 1, int(p.y) + radius) - std::max(0, int(p.y) - radius) + 1);
        blocked(region).setTo(255);
        if (points.size() >= static_cast<size_t>(maxPoints))
            break;
    }
    return points;
}
cv::Mat sampleSuperPointDescriptors(const cv::Mat& coarse, cv::Size size,
                                    const std::vector<cv::Point2f>& pixels) {
    if (coarse.dims != 4 || coarse.type() != CV_32F || !coarse.isContinuous() ||
        coarse.size[0] != 1 || coarse.size[1] != 256 || size.width < 8 || size.height < 8 ||
        coarse.size[2] * 8 != size.height || coarse.size[3] * 8 != size.width ||
        !cv::checkRange(coarse))
        throw std::runtime_error("Invalid SuperPoint descriptor tensor");
    const int h = coarse.size[2], w = coarse.size[3], plane = h * w;
    // Cell-major storage keeps each 256-vector contiguous during normalization and sampling.
    // The channel summation and interpolation order stay identical to the NCHW implementation.
    cv::Mat channels(256, plane, CV_32F, const_cast<float*>(coarse.ptr<float>())), normalized;
    cv::transpose(channels, normalized);
    for (int i = 0; i < plane; ++i) {
        auto* row = normalized.ptr<float>(i);
        double norm = 0;
        for (int c = 0; c < 256; ++c)
            norm += double(row[c]) * row[c];
        norm = std::sqrt(norm);
        for (int c = 0; c < 256; ++c)
            row[c] = norm > 1e-12 ? row[c] / norm : 0;
    }
    cv::Mat result(int(pixels.size()), 256, CV_32F, cv::Scalar(0));
    for (size_t i = 0; i < pixels.size(); ++i) {
        const auto p = pixels[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.y < 0 || p.x >= size.width ||
            p.y >= size.height)
            throw std::runtime_error("Descriptor sample outside image");
        // Original demo: grid=(2*x/W-1,2*y/H-1), historical grid_sample align_corners=True.
        const double x = double(p.x) / size.width * (w - 1),
                     y = double(p.y) / size.height * (h - 1);
        const int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, w - 1),
                  y1 = std::min(y0 + 1, h - 1);
        const double dx = x - x0, dy = y - y0;
        auto* row = result.ptr<float>(int(i));
        double norm = 0;
        const auto* p00 = normalized.ptr<float>(y0 * w + x0);
        const auto* p01 = normalized.ptr<float>(y0 * w + x1);
        const auto* p10 = normalized.ptr<float>(y1 * w + x0);
        const auto* p11 = normalized.ptr<float>(y1 * w + x1);
        for (int c = 0; c < 256; ++c) {
            row[c] = (1 - dy) * ((1 - dx) * p00[c] + dx * p01[c]) +
                     dy * ((1 - dx) * p10[c] + dx * p11[c]);
            norm += double(row[c]) * row[c];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-12)
            for (int c = 0; c < 256; ++c)
                row[c] /= norm;
    }
    return result;
}
struct SuperPointDetector::Impl {
    cv::dnn::Net net;
#ifdef SUPERVO_WITH_TENSORRT
    std::unique_ptr<trtEngine> trt;
    cv::Size trtInputSize;
    cv::Mat trtImage(const cv::Mat& image) const {
        if (image.size() != trtInputSize)
            throw std::runtime_error("Image dimensions disagree with configured TensorRT input");
        cv::Mat padded;
        cv::copyMakeBorder(image, padded, 0, (8 - image.rows % 8) % 8, 0, (8 - image.cols % 8) % 8,
                           cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, 0);
        return padded;
    }
#endif
};
SuperPointDetector::SuperPointDetector(const std::string& weights, const std::string& backend,
                                       const std::string& engine, cv::Size size)
    : impl_(new Impl) {
    if (backend == "tensorrt") {
#ifdef SUPERVO_WITH_TENSORRT
        if (engine.empty() || size.width < 8 || size.height < 8)
            throw std::runtime_error(
                "TensorRT requires an engine and image dimensions at least 8x8");
        impl_->trtInputSize = size;
        impl_->trt.reset(
            new trtEngine((size.height + 7) / 8 * 8, (size.width + 7) / 8 * 8, 1, engine));
        return;
#else
        (void)engine;
        (void)size;
        throw std::runtime_error(
            "TensorRT backend was not enabled at build time; use --backend cpu");
#endif
    }
    if (backend != "cpu")
        throw std::runtime_error("Unknown feature backend: " + backend);
    std::ifstream input(weights);
    if (!input)
        throw std::runtime_error("Cannot open SuperPoint weights: " + weights);
    int blobs = 0;
    input >> blobs;
    if (blobs != 24)
        throw std::runtime_error("Expected 24 SuperPoint weight blobs");
    std::unordered_map<std::string, std::vector<float>> tensors;
    for (int i = 0; i < blobs; ++i) {
        std::string name;
        size_t count = 0;
        input >> name >> std::dec >> count;
        if (!input || count == 0 || count > 10000000 || tensors.count(name))
            throw std::runtime_error("Malformed weights");
        auto& values = tensors[name];
        values.resize(count);
        for (auto& value : values) {
            uint32_t bits;
            input >> std::hex >> bits;
            if (!input)
                throw std::runtime_error("Truncated weights");
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value))
                throw std::runtime_error("Non-finite weights");
        }
    }
    int previous = 0;
    auto layer = [&](const std::string& name, const std::string& type,
                     cv::dnn::LayerParams params) {
        int id = impl_->net.addLayer(name, type, params);
        impl_->net.connect(previous, 0, id, 0);
        previous = id;
    };
    auto conv = [&](const std::string& name, int in, int out, int kernel, bool relu) {
        cv::dnn::LayerParams p;
        p.set("kernel_size", kernel);
        p.set("pad", kernel / 2);
        p.set("num_output", out);
        p.set("bias_term", true);
        auto& w = tensors.at(name + ".weight");
        auto& b = tensors.at(name + ".bias");
        if (w.size() != size_t(out * in * kernel * kernel) || b.size() != size_t(out))
            throw std::runtime_error("Incorrect weight shape: " + name);
        int dims[] = {out, in, kernel, kernel};
        p.blobs = {cv::Mat(4, dims, CV_32F, w.data()).clone(), cv::Mat(b).clone()};
        layer(name, "Convolution", p);
        if (relu)
            layer(name + "_relu", "ReLU", {});
    };
    auto pool = [&](const std::string& name) {
        cv::dnn::LayerParams p;
        p.set("pool", "MAX");
        p.set("kernel_size", 2);
        p.set("stride", 2);
        layer(name, "Pooling", p);
    };
    impl_->net.setInputsNames({"image"});
    conv("conv1a", 1, 64, 3, true);
    conv("conv1b", 64, 64, 3, true);
    pool("pool1");
    conv("conv2a", 64, 64, 3, true);
    conv("conv2b", 64, 64, 3, true);
    pool("pool2");
    conv("conv3a", 64, 128, 3, true);
    conv("conv3b", 128, 128, 3, true);
    pool("pool3");
    conv("conv4a", 128, 128, 3, true);
    conv("conv4b", 128, 128, 3, true);
    const int shared = previous;
    conv("convPa", 128, 256, 3, true);
    conv("convPb", 256, 65, 1, false);
    previous = shared;
    conv("convDa", 128, 256, 3, true);
    conv("convDb", 256, 256, 1, false);
    impl_->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    impl_->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}
SuperPointDetector::~SuperPointDetector() = default;
cv::Mat SuperPointDetector::inferLogits(const cv::Mat& image) {
#ifdef SUPERVO_WITH_TENSORRT
    if (impl_->trt)
        return impl_->trt->infer(impl_->trtImage(image));
#endif
    auto normalized = prepareSuperPointImage(image, image.size());
    cv::copyMakeBorder(normalized, normalized, 0, (8 - image.rows % 8) % 8, 0,
                       (8 - image.cols % 8) % 8, cv::BORDER_CONSTANT, 0);
    impl_->net.setInput(cv::dnn::blobFromImage(normalized));
    return impl_->net.forward("convPb").clone();
}
SuperPointOutput SuperPointDetector::infer(const cv::Mat& image) {
#ifdef SUPERVO_WITH_TENSORRT
    if (impl_->trt)
        return impl_->trt->inferOutputs(impl_->trtImage(image));
#endif
    auto normalized = prepareSuperPointImage(image, image.size());
    cv::copyMakeBorder(normalized, normalized, 0, (8 - image.rows % 8) % 8, 0,
                       (8 - image.cols % 8) % 8, cv::BORDER_CONSTANT, 0);
    impl_->net.setInput(cv::dnn::blobFromImage(normalized));
    std::vector<cv::Mat> outputs;
    impl_->net.forward(outputs, std::vector<cv::String>{"convPb", "convDb"});
    return {outputs[0].clone(), outputs[1].clone()};
}
SuperPointFeatures SuperPointDetector::extract(const cv::Mat& image, float threshold,
                                               int maxPoints) {
    const auto output = infer(image);
    auto pixels = decodeSuperPoint(output.logits, image.size(), threshold, maxPoints);
    auto descriptors = sampleSuperPointDescriptors(
        output.descriptors, {output.descriptors.size[3] * 8, output.descriptors.size[2] * 8},
        pixels);
    return {std::move(pixels), std::move(descriptors)};
}
std::vector<cv::Point2f> SuperPointDetector::detect(const cv::Mat& image, float threshold,
                                                    int maxPoints) {
    return decodeSuperPoint(inferLogits(image), image.size(), threshold, maxPoints);
}
}
