#ifndef TRACKER_H
#define TRACKER_H

#include <trtEngine.h>
#include <mapDrawer.h>
#include <frame.h>
#include <map.h>
#include <optimizer.h>
#include <mapPoint.h>
#include <dataLoader.h>

#include <iostream>
#include <chrono>
#include <opencv4/opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <opencv4/opencv2/core/eigen.hpp>
#include <thread>
#include <sophus/se3.hpp>

namespace superVO
{
    class tracker
    {
    private:
        std::shared_ptr<frame> mFrameCore;
        std::shared_ptr<keyFrame> mKeyFrameCore;
        std::shared_ptr<dataLoader> mDataLoader;
        std::shared_ptr<optimizer> mOptimizer;
        std::shared_ptr<map> mMap;
        std::shared_ptr<trtEngine> mTrtEngine;
        std::shared_ptr<mapDrawer> mMapDrawer;

        std::thread mTrackerThread;
        std::condition_variable mTrack;
        std::atomic<bool> mRunningStatus;
        std::mutex mMutex;

        std::string mInfoType = "[TRACKER]-->";

        std::string mWindowName = "superVO";

        std::vector<Eigen::Isometry3d> mFrameSE3;
        std::ofstream mFout;

    public:
        tracker(std::shared_ptr<trtEngine> pEngine, 
                std::shared_ptr<dataLoader> pLoader,
                std::shared_ptr<frame> pFrame,
                std::shared_ptr<keyFrame>pKeyFrame,
                std::shared_ptr<map> pMap,
                std::shared_ptr<optimizer> pOptimizer,
                std::shared_ptr<mapDrawer> pDrawer);
        void initialize();
        void track();
        void trackFrame();
        void trackKeyFrame(std::string pLastFrame, std::string pCurFrame, std::string pCurFrameRight);
        void waitTime(double lastTime, double curTime, double trackTime);
        ~tracker();
    };
    
}

#endif