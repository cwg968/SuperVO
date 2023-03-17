#ifndef TRTENGINE_H
#define TRTENGINE_H

#include <iostream>
#include <map>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv4/opencv2/opencv.hpp>
#include <torch/torch.h>


namespace superVO
{
    class trtEngine
    {
    private:
        int INPUT_H;
        int INPUT_W;
        int MAX_BATCHSIZE;
        const char* INPUT_BLOB_NAME = "data";
        const char* semi_BLOB_NAME = "semi";
        const char* desc_BLOB_NAME = "desc";
        bool isTRTEngine = false;
        const int cell = 8;
        const int borderRemove = 4;
        const float confThresh = 0.05;
        const int nmsDist = 4;
        
        nvinfer1::IExecutionContext* context;
        std::string engineFile;  

        std::string mInfoType = "[TRTENGINE]-->";
        
    public:
        trtEngine();
        trtEngine(int H, int W, int maxBatchsize, std::string engineFile);
        trtEngine& operator=(trtEngine &t);

        std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

        nvinfer1::IHostMemory* createSuperpointEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, std::string weightsPath);

        void buildTRTEngine(std::string weightsPath);

        nvinfer1::IExecutionContext* loadTRTEngine();

        void doInference(nvinfer1::IExecutionContext& context, float* input, float* semiOutput, float* descOutput, int batchSize);

        std::vector<cv::Mat> reshape(float semi[], float desc[]);

        void getImageData(cv::Mat &image, float* data);

        void getPoints(cv::Mat &image, std::vector<cv::Point3f> &ppts, torch::Tensor &pdesc);

        std::vector<cv::Point3f> nmsFast(std::vector<cv::Point3f> &pts, int H, int W);

        ~trtEngine();

    };

    void getImageHW(std::string imageFile, int* H, int* W);

}

#endif