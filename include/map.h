#ifndef MAP_H
#define MAP_H

#include <mapPoint.h>
#include <unordered_map>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>

namespace superVO
{
    class map
    {
        private:
            std::unordered_map<unsigned long, mapPoint*> mapPoints_;
            std::unordered_map<unsigned long, mapPoint*> activeMapPoints_;
            std::unordered_map<unsigned long, kFrame_*> mapFrames_;
            std::unordered_map<unsigned long, kFrame_*> activeMapFrames_;

            int maxActiveKFrames = 3;

            Eigen::Matrix3d mK;
            Sophus::SE3d mP;
            
            std::mutex mMutex;
            
            std::string mInfoType = "[MAP]-->";
        public:
            map();
            map(cv::Mat k);
            std::unordered_map<unsigned long, mapPoint*> getActiveMapPoints();
            void insertMapPoint(mapPoint* mp);
            std::unordered_map<unsigned long, kFrame_*> getActiveKeyFrames();
            void insertMapFrame(kFrame_* mkf);
            Eigen::Matrix3d K();
            Sophus::SE3d pose();
            ~map();
    };
}


#endif