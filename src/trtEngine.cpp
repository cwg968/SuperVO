#include <iostream>
#include <fstream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <trtEngine.h>
#include <logging.h>

// 实现argsort功能
std::vector<int> argsort(const std::vector<cv::Point3f>& array)
{
	const int array_len(array.size());
	std::vector<int> array_index(array_len, 0);
	for (int i = 0; i < array_len; ++i)
		array_index[i] = i;

	std::sort(array_index.begin(), array_index.end(),
		[&array](int pos1, int pos2) {return (-array[pos1].z < -array[pos2].z);});

	return array_index;
}

static Logger gLogger;

#define TCHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

namespace superVO
{
    trtEngine::trtEngine(){}

    trtEngine::trtEngine(int H, int W, int maxBatchsize, std::string trtEngineFile) : INPUT_H(H), INPUT_W(W), MAX_BATCHSIZE(maxBatchsize), engineFile(trtEngineFile)
    {
        std::cout << "trtEngine object is being created!" << std::endl;
    }

    trtEngine& trtEngine::operator=(trtEngine &t)
    {
        INPUT_H = t.INPUT_H;
        INPUT_W = t.INPUT_W;
        MAX_BATCHSIZE = t.MAX_BATCHSIZE;
        engineFile = t.engineFile;
        return *this;
    }


    trtEngine::~trtEngine()
    {
        std::cout << "trtEngine object is being deleted!" << std::endl;
    }


    std::map<std::string, nvinfer1::Weights> trtEngine::loadWeights(const std::string file)
    {
        std::cout << "Loading weights: " << file << std::endl;
        std::map<std::string, nvinfer1::Weights> weightMap;

        // Open weights file
        std::ifstream input(file);
        assert(input.is_open() && "Unable to load weight file.");

        // Read number of weight blobs
        int32_t count;
        input >> count;
        assert(count > 0 && "Invalid weight map file.");

        while (count--)
        {
            nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
            uint32_t size;

            // Read name and type of blob
            std::string name;
            input >> name >> std::dec >> size;
            wt.type = nvinfer1::DataType::kFLOAT;

            // Load blob
            uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x)
            {
                input >> std::hex >> val[x];
            }
            wt.values = val;

            wt.count = size;
            weightMap[name] = wt;
        }

        return weightMap;
    }

    nvinfer1::IHostMemory* trtEngine::createSuperpointEngine(unsigned int maxBatchSize, nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, nvinfer1::DataType dt, std::string weightsPath)
    {
        const int c1 = 64;
        const int c2 = 64;
        const int c3 = 128;
        const int c4 = 128;
        const int c5 = 256;
        const int d1 = 256;

        nvinfer1::INetworkDefinition* superpoint = builder->createNetworkV2(0U);
        nvinfer1::ITensor* inputData = superpoint->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims4{1, 1, INPUT_H, INPUT_W});
        assert(inputData);

        std::map<std::string, nvinfer1::Weights> weightMap = loadWeights(weightsPath);
        // shared encoder
        nvinfer1::IConvolutionLayer* conv1a = superpoint->addConvolutionNd(*inputData, c1, nvinfer1::DimsHW{3, 3}, weightMap["conv1a.weight"], weightMap["conv1a.bias"]);
        conv1a->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv1a->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv1a);

        nvinfer1::IActivationLayer* relu1a = superpoint->addActivation(*conv1a->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu1a);

        nvinfer1::IConvolutionLayer* conv1b = superpoint->addConvolutionNd(*relu1a->getOutput(0), c1, nvinfer1::DimsHW{3, 3}, weightMap["conv1b.weight"], weightMap["conv1b.bias"]);
        conv1b->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv1b->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv1b);
        
        nvinfer1::IActivationLayer* relu1b = superpoint->addActivation(*conv1b->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu1b);

        nvinfer1::IPoolingLayer* pool1 = superpoint->addPoolingNd(*relu1b->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
        pool1->setStrideNd(nvinfer1::DimsHW{2, 2});
        assert(pool1);

        nvinfer1::IConvolutionLayer* conv2a = superpoint->addConvolutionNd(*pool1->getOutput(0), c2, nvinfer1::DimsHW{3, 3}, weightMap["conv2a.weight"], weightMap["conv2a.bias"]);
        conv2a->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv2a->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv2a);

        nvinfer1::IActivationLayer* relu2a = superpoint->addActivation(*conv2a->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu2a);

        nvinfer1::IConvolutionLayer* conv2b = superpoint->addConvolutionNd(*relu2a->getOutput(0), c2, nvinfer1::DimsHW{3, 3}, weightMap["conv2b.weight"], weightMap["conv2b.bias"]);
        conv2b->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv2b->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv2b);

        nvinfer1::IActivationLayer* relu2b = superpoint->addActivation(*conv2b->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu2b);

        nvinfer1::IPoolingLayer* pool2 = superpoint->addPoolingNd(*relu2b->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
        pool2->setStrideNd(nvinfer1::DimsHW{2, 2});
        assert(pool2);

        nvinfer1::IConvolutionLayer* conv3a = superpoint->addConvolutionNd(*pool2->getOutput(0), c3, nvinfer1::DimsHW{3, 3}, weightMap["conv3a.weight"], weightMap["conv3a.bias"]);
        conv3a->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv3a->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv3a);

        nvinfer1::IActivationLayer* relu3a = superpoint->addActivation(*conv3a->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu3a);

        nvinfer1::IConvolutionLayer* conv3b = superpoint->addConvolutionNd(*relu3a->getOutput(0), c3, nvinfer1::DimsHW{3, 3}, weightMap["conv3b.weight"], weightMap["conv3b.bias"]);
        conv3b->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv3b->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv3b);

        nvinfer1::IActivationLayer* relu3b = superpoint->addActivation(*conv3b->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu3b);

        nvinfer1::IPoolingLayer* pool3 = superpoint->addPoolingNd(*relu3b->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{2, 2});
        pool3->setStrideNd(nvinfer1::DimsHW{2, 2});
        assert(pool3);

        nvinfer1::IConvolutionLayer* conv4a = superpoint->addConvolutionNd(*pool3->getOutput(0), c4, nvinfer1::DimsHW{3, 3}, weightMap["conv4a.weight"], weightMap["conv4a.bias"]);
        conv4a->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv4a->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv4a);

        nvinfer1::IActivationLayer* relu4a = superpoint->addActivation(*conv4a->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu4a);

        nvinfer1::IConvolutionLayer* conv4b = superpoint->addConvolutionNd(*relu4a->getOutput(0), c4, nvinfer1::DimsHW{3, 3}, weightMap["conv4b.weight"], weightMap["conv4b.bias"]);
        conv4b->setStrideNd(nvinfer1::DimsHW{1, 1});
        conv4b->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(conv4b);

        nvinfer1::IActivationLayer* relu4b = superpoint->addActivation(*conv4b->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relu4b);

        // detector head
        nvinfer1::IConvolutionLayer* convpa = superpoint->addConvolutionNd(*relu4b->getOutput(0), c5, nvinfer1::DimsHW(3, 3), weightMap["convPa.weight"], weightMap["convPa.bias"]);
        convpa->setStrideNd(nvinfer1::DimsHW{1, 1});
        convpa->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(convpa);

        nvinfer1::IActivationLayer* relupa = superpoint->addActivation(*convpa->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(relupa);

        nvinfer1::IConvolutionLayer* convpb = superpoint->addConvolutionNd(*relupa->getOutput(0), 65, nvinfer1::DimsHW(1, 1), weightMap["convPb.weight"], weightMap["convPb.bias"]);
        convpb->setStrideNd(nvinfer1::DimsHW{1, 1});
        assert(convpb);

        convpb->getOutput(0)->setName(semi_BLOB_NAME);
        superpoint->markOutput(*convpb->getOutput(0));

        // descriptor head
        nvinfer1::IConvolutionLayer* convda = superpoint->addConvolutionNd(*relu4b->getOutput(0), c5, nvinfer1::DimsHW{3, 3}, weightMap["convDa.weight"], weightMap["convDa.bias"]);
        convda->setStrideNd(nvinfer1::DimsHW{1, 1});
        convda->setPaddingNd(nvinfer1::DimsHW{1, 1});
        assert(convda);

        nvinfer1::IActivationLayer* reluda = superpoint->addActivation(*convda->getOutput(0), nvinfer1::ActivationType::kRELU);
        assert(reluda);

        nvinfer1::IConvolutionLayer* convdb = superpoint->addConvolutionNd(*reluda->getOutput(0), d1, nvinfer1::DimsHW{1, 1}, weightMap["convDb.weight"], weightMap["convDb.bias"]);
        convdb->setStrideNd(nvinfer1::DimsHW{1, 1});
        assert(convdb);

        convdb->getOutput(0)->setName(desc_BLOB_NAME);
        superpoint->markOutput(*convdb->getOutput(0));

        // build engine
        builder->setMaxBatchSize(maxBatchSize);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
        nvinfer1::IHostMemory *engine = builder->buildSerializedNetwork(*superpoint, *config);
        return engine;
    }


    void trtEngine::buildTRTEngine(std::string weightsPath)
    {
        nvinfer1::IHostMemory* modelStream{nullptr};

        std::cout << mInfoType << "Start building TensorRT engine!" <<std::endl;

        // Create builder
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        // Create model to populate the network, then set the outputs and create an engine
        nvinfer1::IHostMemory* plan = createSuperpointEngine(MAX_BATCHSIZE, builder, config, nvinfer1::DataType::kFLOAT, weightsPath);
        assert(plan != nullptr);

        nvinfer1::IRuntime* runtime  = nvinfer1::createInferRuntime(gLogger);
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan->data(), plan->size(), {nullptr});

        // Serialize the engine
        modelStream = engine->serialize();

        // Close everything down
        delete engine;
        delete builder;
        delete config;
        
        assert(modelStream != nullptr);

        std::ofstream p(engineFile, std::ios::binary);
        if(!p)
        {
            std::cerr << mInfoType << "could not open plan output file" << std::endl;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << mInfoType << "TensorRT engine has builed!" << std::endl;

        delete modelStream;
    }

    nvinfer1::IExecutionContext* trtEngine::loadTRTEngine()
    {
        char* trtModelStream{nullptr};
        size_t size{0};
        std::ifstream file(engineFile, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
        
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        assert(runtime != nullptr);
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
        assert(engine != nullptr);
        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        assert(context != nullptr);

        delete []trtModelStream;
        return context;
    }

    void trtEngine::doInference(nvinfer1::IExecutionContext& context, float* input, float* semiOutput, float* descOutput, int batchSize)
    {
        const nvinfer1::ICudaEngine& engine = context.getEngine();
        assert(engine.getNbBindings() == 3);
        void* buffers[3];

        const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
        const int semiIndex = engine.getBindingIndex(semi_BLOB_NAME);
        const int descIndex = engine.getBindingIndex(desc_BLOB_NAME);

        // Create GPU buffers on device
        TCHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
        TCHECK(cudaMalloc(&buffers[semiIndex], batchSize * 65 * INPUT_H / 8 * INPUT_W / 8 * sizeof(float)));
        TCHECK(cudaMalloc(&buffers[descIndex], batchSize * 256 * INPUT_H / 8 * INPUT_W / 8 * sizeof(float)));
        
        // Create stream
        cudaStream_t stream;
        TCHECK(cudaStreamCreate(&stream));

        TCHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context.enqueue(batchSize, buffers, stream, nullptr);
        TCHECK(cudaMemcpyAsync(semiOutput, buffers[semiIndex], batchSize * 65 * INPUT_H / 8 * INPUT_W /8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        TCHECK(cudaMemcpyAsync(descOutput, buffers[descIndex], batchSize * 256 * INPUT_H / 8 * INPUT_W /8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        cudaStreamDestroy(stream);
        TCHECK(cudaFree(buffers[inputIndex]));
        TCHECK(cudaFree(buffers[semiIndex]));
        TCHECK(cudaFree(buffers[descIndex]));
    }

    // 8.9
    std::vector<cv::Mat> trtEngine::reshape(float semi[], float desc[])
    {
        cv::Mat semiTmp(65, INPUT_H * INPUT_W / 64, CV_32FC1, semi);
        cv::Mat descTmp(256, INPUT_H * INPUT_W / 64, CV_32FC1, desc);
        cv::Mat descNormDim0(1, INPUT_H * INPUT_W / 64, CV_32FC1);
        for(int i = 0; i < descTmp.rows; i++)
        {
            cv::Mat descRowTmp;
            cv::pow(descTmp.rowRange(i, i + 1), 2, descRowTmp);
            descNormDim0 += descRowTmp;
        }
        cv::sqrt(descNormDim0, descNormDim0);
        for(int i = 0; i < descTmp.rows; i++)
        {
            cv::divide(descTmp.rowRange(i, i + 1), descNormDim0, descTmp.rowRange(i, i + 1));
        }

        std::vector<cv::Mat> output;
        output.emplace_back(semiTmp);
        output.emplace_back(descTmp);
        return output;
    }
    // 8.9
    void trtEngine::getImageData(cv::Mat &image, float* data)
    {
        cv::Mat imageGray;
        if(image.channels() == 3)
        {
            cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
        }
        
        uchar* inputData;
        if(imageGray.isContinuous())
        {
            inputData = imageGray.data;
        }
        for(int i = 0; i < INPUT_H * INPUT_W; i++)
        {
            data[i] = inputData[i] / 255.0;
        }
    }

    std::vector<cv::Point3f> trtEngine::nmsFast(std::vector<cv::Point3f> &pts, int H, int W)
    {
        cv::Mat grid = cv::Mat::zeros(H, W, CV_8SC1);
        cv::Mat inds = cv::Mat::zeros(H, W, CV_32SC1);
        
        //argsort for pts.z
        std::vector<int> inds1 = argsort(pts);

        std::vector<cv::Point3f> corners;
        for(int i = 0; i < inds1.size(); i++)
        {
            corners.emplace_back(cv::Point3f(round(pts[inds1[i]].x), round(pts[inds1[i]].y), pts[inds1[i]].z));
        }

        for(int i = 0; i < corners.size(); i++)
        {
            grid.at<char>(corners[i].y, corners[i].x) = 1;
            inds.at<int>(corners[i].y, corners[i].x) = i;
        }

        int pad = nmsDist;
        cv::Mat gridPad = cv::Mat::zeros(H + 2 * pad, W + 2 * pad, CV_8SC1);
        grid.copyTo(gridPad(cv::Rect(pad, pad, W, H)));

        for(int i = 0; i < corners.size(); i++)
        {
            cv::Point2i pt(corners[i].x + pad, corners[i].y + pad);
            if(gridPad.at<char>(pt.y, pt.x) == 1)
            {
                for(int j = pt.y - pad; j < pt.y + pad + 1; j++)
                {
                    for(int k = pt.x - pad; k < pt.x + pad + 1; k++)
                    {
                        gridPad.at<char>(j, k) = 0;
                    }
                }
                gridPad.at<char>(pt.y, pt.x) = -1;
            }
        }

        std::vector<cv::Point2i> keep;
        for(int i = 0; i < gridPad.rows; i++)
        {
            for(int j = 0; j < gridPad.cols; j++)
            {
                if(gridPad.at<char>(i, j) == -1)
                    keep.emplace_back(cv::Point2i(i - pad, j - pad));
            }
        }

        std::vector<int> indsKeep;
        for(int i = 0; i < keep.size(); i++)
        {
            indsKeep.emplace_back(inds.at<int>(keep[i].x, keep[i].y));
        }

        std::vector<cv::Point3f> out;
        for(int i = 0; i < indsKeep.size(); i++)
        {
            out.emplace_back(corners[indsKeep[i]]);
        }

        std::vector<int> inds2 = argsort(out);

        std::vector<cv::Point3f> out1;
        for(int i = 0; i < inds2.size(); i++)
        {
            out1.emplace_back(out[inds2[i]]);
        }

        return out1;
    }


    void trtEngine::getPoints(cv::Mat &image, std::vector<cv::Point3f> &ppts, torch::Tensor &pdesc)
    {
        float* inputData = new float[INPUT_H * INPUT_W];
        float* semi = new float[65 * INPUT_H / 8 * INPUT_W / 8];
        float* desc = new float[256 * INPUT_H / 8 * INPUT_W / 8];

        getImageData(image, inputData);

        if(!isTRTEngine)
        {
            context = loadTRTEngine();
            isTRTEngine = true;
        }

        auto startTime = std::chrono::steady_clock::now();
        doInference(*context, inputData, semi, desc, 1);
        auto endTime = std::chrono::steady_clock::now();
        auto timeUsed =  std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        std::cout << mInfoType << "Inference time used: " << timeUsed.count() << std::endl;

        auto t1 = std::chrono::steady_clock::now();    
        std::vector<cv::Mat> output = reshape(semi, desc);
        auto t2 = std::chrono::steady_clock::now();
        auto timeUsed1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        
        cv::Mat A;

        cv::Mat semiTensor = output[0];
        cv::Mat descTensor = output[1].t();

        cv::exp(semiTensor, semiTensor);


        cv::Mat noDust(64, INPUT_H * INPUT_W / 64, CV_32FC1);
        cv::Mat constant(1, INPUT_H * INPUT_W / 64, CV_32FC1);
        constant = 0.00001;

        cv::Mat semiTensorRowSumDim0(1, INPUT_H * INPUT_W / 64, CV_32FC1);
        semiTensorRowSumDim0 = 0;
        for(int i = 0; i < semiTensor.rows; i++)
        {
            semiTensorRowSumDim0 += semiTensor.rowRange(i, i + 1);
        }
        semiTensorRowSumDim0 += constant;
        for(int i = 0; i < noDust.rows; i++)
        {
            cv::divide(semiTensor.rowRange(i, i + 1), semiTensorRowSumDim0, noDust.rowRange(i, i+ 1));
        }
        
        cv::Mat noDustShuffle = noDust.t();

        // std::cout << noDustShuffle.rowRange(0, 1) << std::endl;

        cv::Mat heatMap(INPUT_H, INPUT_W,CV_32FC1);
        int heatMapCellRow = 0;
        int heatMapCellCol = 0;
        for(int i = 0; i < INPUT_H / 8 * INPUT_W / 8; i++)
        {
            cv::Mat noDustShuffleRow = noDustShuffle.rowRange(i, i + 1);
            cv::Mat noDustShuffleRow8x8 = noDustShuffleRow.reshape(0, 8);
            heatMapCellCol = i % (INPUT_W / 8);
            noDustShuffleRow8x8.copyTo(heatMap(cv::Rect(cell * heatMapCellCol, cell * heatMapCellRow, cell, cell)));
            if((i + 1) * cell % INPUT_W == 0)
            {
                heatMapCellRow++;
            }
            heatMapCellCol++;
        }

        // std::cout << heatMap.rowRange(0, 1) << std::endl;

        std::vector<cv::Point3f> pts;
        for(int i = 0; i < heatMap.rows; i++)
        {
            for(int j = 0; j < heatMap.cols; j++)
            {
                if(heatMap.at<float >(i, j) > confThresh)
                {
                    pts.emplace_back(cv::Point3f(j, i, heatMap.at<float>(i, j)));
                }
            }
        }

        std::vector<cv::Point3f> nmsPts;
        auto t11 = std::chrono::steady_clock::now();
        nmsPts = nmsFast(pts, INPUT_H, INPUT_W);
        auto t22 = std::chrono::steady_clock::now();
        auto nmsTime = std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t11);

        // Remove points along border. 移除沿边界的点。
        int bord = borderRemove;
        std::vector<cv::Point3f> ptsRemoveBorder;

        for(int i = 0; i < nmsPts.size(); i++)
        {
            if(nmsPts[i].x >= bord || nmsPts[i].x < INPUT_W - bord || nmsPts[i].y >= bord || nmsPts[i].y < INPUT_H - bord)
            {
                ptsRemoveBorder.emplace_back(nmsPts[i]);
            }
        }


        // --- Process descriptor.
        // 处理描述子
        // torch::Tensor samp_pts = torch::from_blob(ptsRemoveBorder.data(), {(int)ptsRemoveBorder.size(), 3}, torch::kFloat);
        // samp_pts = samp_pts.index_select(1, torch::tensor({0, 1}));
        // samp_pts.select(1, 0) = (samp_pts.select(1, 0) / (INPUT_W / 2)) - 1;
        // samp_pts.select(1, 1) = (samp_pts.select(1, 1) / (INPUT_H / 2)) - 1;
        // samp_pts = samp_pts.transpose(0, 1).contiguous();
        // samp_pts = samp_pts.view({1, 1, -1, 2});
        // samp_pts = torch::nan_to_num(samp_pts);
        
        // torch::Tensor course_desc = torch::from_blob(descTensor.data, {1, 256, INPUT_H / 8, INPUT_W / 8}, torch::kFloat);
        // torch::Tensor desc_t = torch::grid_sampler(course_desc, samp_pts, 0, 0, false);
        // desc_t = desc_t.reshape({256, -1});
        // desc_t = torch::nan_to_num(desc_t);
        // desc_t /= torch::linalg::norm(desc_t, 2, 0, false, c10::nullopt);
        // pdesc = desc_t.clone();

        ppts = ptsRemoveBorder;
        
        delete []inputData;
        delete []semi;
        delete []desc;

    }  
}



void getImageHW(std::string imageFile, int* H, int* W)
{
    cv::Mat img = cv::imread(imageFile, 0);
    *H = img.rows;
    *W = img.cols;
}






