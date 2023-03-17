#include <optimizer.h>
namespace superVO
{
    optimizer::optimizer(std::shared_ptr<map> pmap)
    {
        mmap = pmap;
        mrunningStatus.store(true);
        moptimizeThread = std::thread(std::bind(&optimizer::run, this));
    }

    void optimizer::run()
    {
        int num = 0;
        while (mrunningStatus.load())
        {
            std::unique_lock<std::mutex> lock(mdataMutex);
            mupdate.wait(lock);
            auto activeKeyFrames = mmap->getActiveKeyFrames();
            auto activeMapPoints = mmap->getActiveMapPoints();

            std::cout << mInfoType << "Frames need to optimize: " << activeKeyFrames.size() << std::endl;
            std::cout << mInfoType << "Points need to optimize: " << activeMapPoints.size() << std::endl;

            auto t1 = std::chrono::steady_clock::now();
            optimize(activeKeyFrames, activeMapPoints);
            auto t2 = std::chrono::steady_clock::now();
            std::cout << mInfoType << "Optimize times: " << num++ << " used " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
        }
        
    }

    void optimizer::updateMap()
    {
        std::unique_lock<std::mutex> lock(mdataMutex);
        mupdate.notify_one();
    }

    void optimizer::optimize(std::unordered_map<unsigned long, kFrame_*> &keyframes, std::unordered_map<unsigned long, mapPoint*> &mappoints)
    {
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer spOptimizer;
        spOptimizer.setAlgorithm(solver);

        std::map<unsigned long, VertexPose*> vertices;
        unsigned long maxKfId = 0;
        for(auto keyframe: keyframes)
        {
            auto kf = keyframe.second;
            VertexPose *vPose = new VertexPose();
            vPose->setId(kf->id_);
            vPose->setEstimate(kf->pose_);
            spOptimizer.addVertex(vPose);
            if(kf->id_ > maxKfId)
            {
                maxKfId = kf->id_;
            }
            vertices.insert({kf->id_, vPose});
        }

        std::map<unsigned long, VertexXYZ*> verticesLandmarkes;
        Eigen::Matrix<double, 3, 3> K = mmap->K();
        Sophus::SE3d left_ext = mmap->pose();

        int index = 1;
        float chi2_th = 5.991;
        std::map<EdgeProjection*, superPoint_*> edges_and_features;

        for(auto mapPoint: mappoints)
        {
            unsigned long mapPointId = mapPoint.second->getID();
            std::vector<superPoint_*> observations = mapPoint.second->getObs();
            for(auto ob: observations)
            {
                kFrame_* frame = ob->kf_;
                EdgeProjection *edge = nullptr;
                edge = new EdgeProjection(K, left_ext);
                if(verticesLandmarkes.find(mapPointId) == verticesLandmarkes.end())
                {
                    VertexXYZ *v = new VertexXYZ;
                    v->setEstimate(mapPoint.second->getPose());
                    v->setId(mapPointId + maxKfId + 1);
                    v->setMarginalized(true);
                    verticesLandmarkes.insert({mapPointId, v});
                    spOptimizer.addVertex(v);
                }
                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->id_));
                edge->setVertex(1, verticesLandmarkes.at(mapPointId));
                edge->setMeasurement(Eigen::Vector2d(ob->pt_.x, ob->pt_.y));
                edge->setInformation(Eigen::Matrix2d::Identity());
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, ob});

                spOptimizer.addEdge(edge);
                index++;
            }
        }

        spOptimizer.initializeOptimization();
        spOptimizer.optimize(10);

        int cnt_outLier = 0;
        int cnt_inLier = 0;
        int iteration = 0;
        while (iteration < 5)
        {
            cnt_outLier = 0;
            cnt_inLier = 0;
            for(auto ef: edges_and_features)
            {
                if(ef.first->chi2() > chi2_th)
                {
                    cnt_outLier++;
                }
                else
                {
                    cnt_inLier++;
                }
            }
            double inLier_ration = cnt_inLier / double(cnt_inLier + cnt_outLier);
            if(inLier_ration > 0.5)
            {
                break;
            }
            else
            {
                chi2_th *= 2;
                iteration++;
            }
        }

        for(auto v: vertices)
        {
            keyframes.at(v.first)->setPose(v.second->estimate());
        }
        for(auto &v: verticesLandmarkes)
        {
            mappoints.at(v.first)->setPose(v.second->estimate());
        }
    }

    void optimizer::stop()
    {
        mrunningStatus.store(false);
        mupdate.notify_one();
        moptimizeThread.join();
    }

    optimizer::~optimizer()
    {
    }
}