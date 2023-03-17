#include <tracker.h>

namespace superVO
{
    tracker::tracker(std::shared_ptr<trtEngine> pEngine, 
                std::shared_ptr<dataLoader> pLoader,
                std::shared_ptr<frame> pFrame,
                std::shared_ptr<keyFrame>pKeyFrame,
                std::shared_ptr<map> pMap,
                std::shared_ptr<optimizer> pOptimizer,
                std::shared_ptr<mapDrawer> pDrawer)
    {
        mTrtEngine = pEngine;
        mDataLoader = pLoader;
        mFrameCore = pFrame;
        mKeyFrameCore = pKeyFrame;
        mMap = pMap;
        mOptimizer = pOptimizer;
        mMapDrawer = pDrawer;

        mFout.open("trajectory.txt");
                
        initialize();

        mTrackerThread = std::thread(std::bind(&tracker::track, this));
    }

    void tracker::initialize()
    {
        mKeyFrameCore->constractWorldFrame(mDataLoader->nextLeftFrame(), mDataLoader->nextRightFrame(), mDataLoader->getCurFrameTime());
        mKeyFrameCore->setPose(cv::Mat::eye(4, 4, CV_32FC1));
        mKeyFrameCore->upDateMap();
        torch::Tensor lastKFDesc = mKeyFrameCore->getDesc();

        cv::namedWindow(mWindowName, cv::WINDOW_AUTOSIZE);
        cv::imshow(mWindowName, mKeyFrameCore->getDrawnImage());
        cv::waitKey(0);
    }

    void tracker::trackFrame()
    {
        mFrameSE3.emplace_back(mFrameCore->getSE3());
        
        // save pose as tum format
        std::vector<double> currentFramePose = mFrameCore->getPose();
        mFout << std::fixed << currentFramePose[0] << " " << currentFramePose[1] << " " << currentFramePose[2] << " " << currentFramePose[3] << " " << currentFramePose[4] << " " << currentFramePose[5] << " " << currentFramePose[6] << " " << currentFramePose[7] << std::endl;
        std::cout << mInfoType << "Pose: " << std::fixed << currentFramePose[0] << " " << currentFramePose[1] << " " << currentFramePose[2] << " " << currentFramePose[3] << " " << currentFramePose[4] << " " << currentFramePose[5] << " " << currentFramePose[6] << " " << currentFramePose[7] << std::endl;
        cv::imshow(mWindowName, mFrameCore->getDrawnImage());
        cv::waitKey(1);        
    }

    void tracker::track()
    {
        while(true)
        {
            std::string lastFrameFile = mDataLoader->lastFrame();
            std::string currentFrameFile = mDataLoader->nextLeftFrame();
            std::string currentFrameFileRight = mDataLoader->nextRightFrame();
            auto t1 = std::chrono::steady_clock::now();

            mFrameCore->updateFrame(currentFrameFile, mDataLoader->getCurFrameTime());     
            
            std::cout << " / " << mDataLoader->LKTHRESH() * mKeyFrameCore->mkpts.size() << std::endl;
            if(mFrameCore->getGoodPtsNum() >= mDataLoader->LKTHRESH() * mKeyFrameCore->mkpts.size())
            {
                mFrameCore->updateFramePose();
                float reprojectError = mFrameCore->reproject();
                
                auto t2 = std::chrono::steady_clock::now(); 
                std::cout << " / " << mDataLoader->ReprojectThresh() * mKeyFrameCore->getPts3D().size() << std::endl;
                std::cout << mInfoType << "Track frame time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
                
                if(mFrameCore->getPts3D().size() >= mDataLoader->ReprojectThresh() * mKeyFrameCore->getPts3D().size())
                {
                    std::cout << mInfoType << "ReprojectError: " << std::sqrt(reprojectError / mFrameCore->getPts3D().size()) << std::endl;
                    trackFrame();
                }
                else
                {
                    t1 = std::chrono::steady_clock::now(); 
                    trackKeyFrame(lastFrameFile, currentFrameFile, currentFrameFileRight);
                    t2 = std::chrono::steady_clock::now(); 
                    std::cout << mInfoType << "Track keyframe time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;
                }
            }
            else
            {
                t1 = std::chrono::steady_clock::now(); 
                trackKeyFrame(lastFrameFile, currentFrameFile, currentFrameFileRight);
                auto t2 = std::chrono::steady_clock::now(); 
                std::cout << mInfoType << "Track keyframe time used: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << std::endl;                
            }
        }
    }

    void tracker::trackKeyFrame(std::string pLastFrame, std::string pCurFrame, std::string pCurFrameRight)
    {
        mKeyFrameCore->updateKeyFrame(pLastFrame, pCurFrame, pCurFrameRight, mDataLoader->getCurFrameTime());
        
        // estimate keyframe pose
        mKeyFrameCore->updateKeyFramePose(mFrameSE3[mFrameSE3.size() - 1]);
        mKeyFrameCore->upDateMap();

        // optimizer run
        mOptimizer->updateMap();

        // save pose as tum format
        std::vector<double> keyFramePose = mKeyFrameCore->getPose();
        mFout << std::fixed << keyFramePose[0] << " " << keyFramePose[1] << " " << keyFramePose[2] << " " << keyFramePose[3] << " " << keyFramePose[4] << " " << keyFramePose[5] << " " << keyFramePose[6] << " " << keyFramePose[7] << std::endl;
        std::cout << mInfoType << "Pose: " << std::fixed << keyFramePose[0] << " " << keyFramePose[1] << " " << keyFramePose[2] << " " << keyFramePose[3] << " " << keyFramePose[4] << " " << keyFramePose[5] << " " << keyFramePose[6] << " " << keyFramePose[7] << std::endl;

        cv::imshow(mWindowName, mKeyFrameCore->getDrawnImage());
        cv::waitKey(1);
    }
    
    tracker::~tracker()
    {
        if(mTrackerThread.joinable())
        {
            mTrackerThread.join();
        }
    }
}