#include <map.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace superVO
{
    map::map(){}
    map::map(cv::Mat k)
    {
        mK << k.at<double>(0, 0), k.at<double>(0, 1), k.at<double>(0, 2), 
              k.at<double>(1, 0), k.at<double>(1, 1), k.at<double>(1, 2), 
              k.at<double>(2, 0), k.at<double>(2, 1), k.at<double>(2, 2);
        mP = Sophus::SE3d(Eigen::Matrix3d::Identity(), Eigen::Vector3d(0, 0, 0));
    }

    std::unordered_map<unsigned long, mapPoint*> map::getActiveMapPoints()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return activeMapPoints_;
    }

    void map::insertMapPoint(mapPoint* mp)
    {
        if(mapPoints_.find(mp->getID()) == mapPoints_.end())
        {
            mapPoints_.insert(std::make_pair(mp->getID(), mp));
            activeMapPoints_.insert(std::make_pair(mp->getID(), mp));
        }  
    }

    std::unordered_map<unsigned long, kFrame_*> map::getActiveKeyFrames()
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return activeMapFrames_;
    }

    void map::insertMapFrame(kFrame_* mkf)
    {
        if(mapFrames_.find(mkf->id_) == mapFrames_.end())
        {
            mapFrames_.insert(std::make_pair(mkf->id_, mkf));
            activeMapFrames_.insert(std::make_pair(mkf->id_, mkf));
        }
        if(activeMapFrames_.size() > maxActiveKFrames)
        {
            unsigned long frameNeedEraseId = 99999;
            for(auto mf: activeMapFrames_)
            {
                if(mf.second->id_ < frameNeedEraseId)
                {
                    frameNeedEraseId = mf.second->id_;
                }
            }
            auto mapPointNeedEraseIds = activeMapFrames_.find(frameNeedEraseId)->second->getMapPointIds();
            for(auto mpId: mapPointNeedEraseIds)
            {
                activeMapPoints_.erase(mpId);
            }
            activeMapFrames_.erase(frameNeedEraseId);
        }
    }

    Eigen::Matrix3d map::K()
    {
        return mK;
    }

    Sophus::SE3d map::pose()
    {
        return mP;
    }

    map::~map()
    {

    }

}