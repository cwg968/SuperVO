#include <trtEngine.h>
#include <superPoint.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

namespace {
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "TensorRT: " << message << '\n';
    }
};
Logger logger;
void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
template <class T> T* checked(T* value, const std::string& operation) {
    if (!value)
        throw std::runtime_error("TensorRT failed: " + operation);
    return value;
}
using Weights = std::map<std::string, std::vector<float>>;
Weights loadWeights(const std::string& path) {
    std::ifstream file(path);
    int blobs = 0;
    if (!(file >> blobs) || blobs != 24)
        throw std::runtime_error("Expected 24 SuperPoint weight blobs");
    Weights tensors;
    for (int i = 0; i < blobs; ++i) {
        std::string name;
        size_t count = 0;
        if (!(file >> name >> std::dec >> count) || count == 0 || count > 10000000 ||
            tensors.count(name))
            throw std::runtime_error("Malformed SuperPoint weights");
        auto& values = tensors[name];
        values.resize(count);
        for (auto& value : values) {
            uint32_t bits = 0;
            if (!(file >> std::hex >> bits))
                throw std::runtime_error("Truncated SuperPoint weights");
            std::memcpy(&value, &bits, sizeof(value));
            if (!std::isfinite(value))
                throw std::runtime_error("Non-finite SuperPoint weights");
        }
    }
    std::string trailing;
    if (file >> trailing)
        throw std::runtime_error("Unexpected trailing SuperPoint weights");
    return tensors;
}
}

namespace superVO {
struct trtEngine::Impl {
    int height, width;
    std::string path;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream = nullptr;
    std::array<void*, 3> buffers{};
    bool ready = false;
    Impl(int h, int w, std::string file) : height(h), width(w), path(std::move(file)) {}
    ~Impl() {
        // Finish pending work before releasing the context or any buffer, also on error paths.
        if (stream)
            cudaStreamSynchronize(stream);
        context.reset();
        for (auto buffer : buffers)
            if (buffer)
                cudaFree(buffer);
        if (stream)
            cudaStreamDestroy(stream);
    }
    void load() {
        if (ready)
            return;
        // Failed initialization is not retried on partially allocated state.
        if (runtime)
            throw std::runtime_error("TensorRT initialization previously failed");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file || file.tellg() <= 0)
            throw std::runtime_error("Cannot read TensorRT engine: " + path);
        std::vector<char> plan(static_cast<size_t>(file.tellg()));
        file.seekg(0);
        if (!file.read(plan.data(), plan.size()))
            throw std::runtime_error("Truncated TensorRT engine");
        runtime.reset(checked(nvinfer1::createInferRuntime(logger), "create runtime"));
        engine.reset(checked(runtime->deserializeCudaEngine(plan.data(), plan.size()),
                             "deserialize engine"));
        if (engine->getNbIOTensors() != 3)
            throw std::runtime_error("Expected data, semi and desc tensors");
        const std::array<const char*, 3> names{"data", "semi", "desc"};
        const std::array<nvinfer1::Dims4, 3> shapes{nvinfer1::Dims4{1, 1, height, width},
                                                    {1, 65, height / 8, width / 8},
                                                    {1, 256, height / 8, width / 8}};
        for (size_t i = 0; i < names.size(); ++i) {
            const auto dims = engine->getTensorShape(names[i]);
            bool matches = dims.nbDims == 4;
            for (int axis = 0; matches && axis < 4; ++axis)
                matches = dims.d[axis] == shapes[i].d[axis];
            const auto mode =
                i == 0 ? nvinfer1::TensorIOMode::kINPUT : nvinfer1::TensorIOMode::kOUTPUT;
            if (!matches || engine->getTensorIOMode(names[i]) != mode ||
                engine->getTensorDataType(names[i]) != nvinfer1::DataType::kFLOAT ||
                engine->getTensorFormat(names[i]) != nvinfer1::TensorFormat::kLINEAR ||
                engine->getTensorLocation(names[i]) != nvinfer1::TensorLocation::kDEVICE)
                throw std::runtime_error(std::string("Incompatible engine tensor: ") + names[i]);
        }
        context.reset(checked(engine->createExecutionContext(), "create execution context"));
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create CUDA stream");
        for (size_t i = 0; i < names.size(); ++i) {
            size_t count = 1;
            for (int axis = 0; axis < 4; ++axis)
                count *= shapes[i].d[axis];
            cudaCheck(cudaMalloc(&buffers[i], count * sizeof(float)), "allocate tensor");
            if (!context->setTensorAddress(names[i], buffers[i]))
                throw std::runtime_error(std::string("Cannot bind tensor: ") + names[i]);
        }
        ready = true;
    }
};

trtEngine::trtEngine(int height, int width, int maxBatchSize, std::string engineFile)
    : impl_(new Impl(height, width, std::move(engineFile))) {
    if (height < 8 || width < 8 || height % 8 || width % 8 || maxBatchSize != 1 ||
        impl_->path.empty())
        throw std::runtime_error(
            "TensorRT requires batch 1, an engine path and dimensions divisible by eight");
}
trtEngine::~trtEngine() = default;

void trtEngine::buildTRTEngine(const std::string& weightsPath) {
    const auto tensors = loadWeights(weightsPath);
    std::unique_ptr<nvinfer1::IBuilder> builder(
        checked(nvinfer1::createInferBuilder(logger), "create builder"));
    // TensorRT 10 networks always use explicit batch dimensions.
    std::unique_ptr<nvinfer1::INetworkDefinition> network(
        checked(builder->createNetworkV2(0), "create network"));
    std::unique_ptr<nvinfer1::IBuilderConfig> config(
        checked(builder->createBuilderConfig(), "create config"));
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
    // Establish an FP32 baseline before evaluating lower-precision engines separately.
    config->clearFlag(nvinfer1::BuilderFlag::kTF32);
    auto* input = checked(network->addInput("data", nvinfer1::DataType::kFLOAT,
                                            nvinfer1::Dims4{1, 1, impl_->height, impl_->width}),
                          "add input");
    auto weights = [&](const std::string& name, size_t count) {
        const auto& values = tensors.at(name);
        if (values.size() != count)
            throw std::runtime_error("Incorrect weight shape: " + name);
        return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, values.data(),
                                 static_cast<int64_t>(count)};
    };
    auto conv = [&](nvinfer1::ITensor* source, const std::string& name, int in, int out, int kernel,
                    bool relu) {
        auto* layer = checked(
            network->addConvolutionNd(*source, out, nvinfer1::DimsHW{kernel, kernel},
                                      weights(name + ".weight", size_t(in) * out * kernel * kernel),
                                      weights(name + ".bias", out)),
            name);
        layer->setName(name.c_str());
        layer->setPaddingNd(nvinfer1::DimsHW{kernel / 2, kernel / 2});
        auto* result = layer->getOutput(0);
        if (relu)
            result = checked(network->addActivation(*result, nvinfer1::ActivationType::kRELU),
                             name + " ReLU")
                         ->getOutput(0);
        return result;
    };
    auto pool = [&](nvinfer1::ITensor* source) {
        auto* layer = checked(
            network->addPoolingNd(*source, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2}),
            "pool");
        layer->setStrideNd(nvinfer1::DimsHW{2, 2});
        return layer->getOutput(0);
    };
    auto* shared = conv(input, "conv1a", 1, 64, 3, true);
    shared = pool(conv(shared, "conv1b", 64, 64, 3, true));
    shared = conv(shared, "conv2a", 64, 64, 3, true);
    shared = pool(conv(shared, "conv2b", 64, 64, 3, true));
    shared = conv(shared, "conv3a", 64, 128, 3, true);
    shared = pool(conv(shared, "conv3b", 128, 128, 3, true));
    shared = conv(shared, "conv4a", 128, 128, 3, true);
    shared = conv(shared, "conv4b", 128, 128, 3, true);
    auto* semi = conv(conv(shared, "convPa", 128, 256, 3, true), "convPb", 256, 65, 1, false);
    auto* desc = conv(conv(shared, "convDa", 128, 256, 3, true), "convDb", 256, 256, 1, false);
    semi->setName("semi");
    network->markOutput(*semi);
    desc->setName("desc");
    network->markOutput(*desc);
    std::unique_ptr<nvinfer1::IHostMemory> plan(
        checked(builder->buildSerializedNetwork(*network, *config), "build engine"));
    std::ofstream output(impl_->path, std::ios::binary);
    if (!output || !output.write(static_cast<const char*>(plan->data()), plan->size()))
        throw std::runtime_error("Cannot write TensorRT engine: " + impl_->path);
}

cv::Mat trtEngine::infer(const cv::Mat& image) {
    return inferOutputs(image, false).logits;
}
SuperPointOutput trtEngine::inferOutputs(const cv::Mat& image, bool withDescriptors) {
    const auto normalized = prepareSuperPointImage(image, {impl_->width, impl_->height});
    impl_->load();
    const int shape[] = {1, 65, impl_->height / 8, impl_->width / 8};
    SuperPointOutput output;
    output.logits.create(4, shape, CV_32F);
    const int descShape[] = {1, 256, impl_->height / 8, impl_->width / 8};
    if (withDescriptors)
        output.descriptors.create(4, descShape, CV_32F);
    auto& logits = output.logits;
    struct FinishOnError {
        cudaStream_t stream;
        ~FinishOnError() {
            if (stream)
                cudaStreamSynchronize(stream);
        }
    } finish{impl_->stream};
    cudaCheck(cudaMemcpyAsync(impl_->buffers[0], normalized.ptr<float>(),
                              normalized.total() * sizeof(float), cudaMemcpyHostToDevice,
                              impl_->stream),
              "upload image");
    if (!impl_->context->enqueueV3(impl_->stream))
        throw std::runtime_error("TensorRT enqueueV3 failed");
    cudaCheck(cudaMemcpyAsync(logits.ptr<float>(), impl_->buffers[1],
                              logits.total() * sizeof(float), cudaMemcpyDeviceToHost,
                              impl_->stream),
              "download logits");
    if (withDescriptors)
        cudaCheck(cudaMemcpyAsync(output.descriptors.ptr<float>(), impl_->buffers[2],
                                  output.descriptors.total() * sizeof(float),
                                  cudaMemcpyDeviceToHost, impl_->stream),
                  "download descriptors");
    cudaCheck(cudaStreamSynchronize(impl_->stream), "synchronize inference");
    finish.stream = nullptr;
    return output;
}

void trtEngine::getPoints(cv::Mat& image, std::vector<cv::Point3f>& points, cv::Mat& descriptors,
                          float threshold, int maxPoints) {
    const auto output = inferOutputs(image);
    const auto pixels = decodeSuperPoint(output.logits, image.size(), threshold, maxPoints);
    points.clear();
    for (auto pixel : pixels)
        points.emplace_back(pixel.x, pixel.y, 1.f);
    descriptors = sampleSuperPointDescriptors(output.descriptors, image.size(), pixels);
}
}
