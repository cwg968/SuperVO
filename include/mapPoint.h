#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <list>

namespace superVO
{
    struct kFrame_
    {
        kFrame_(){};
        kFrame_(Eigen::Isometry3d se3)
        {
            static unsigned long id = 0;
            id_ = id++;
            Rwk_ = se3.rotation();
            twk_ = se3.translation();
            pose_ = Sophus::SE3d(Eigen::Quaterniond(Rwk_), twk_);
        }

        unsigned long id_;
        Sophus::SE3d pose_;
        Eigen::Matrix3d Rwk_;
        Eigen::Vector3d twk_;
        std::vector<unsigned long> ids_;
        void setPose(Sophus::SE3d pose)
        {
            pose_ = pose;
        }
        void setMapPointIds(std::vector<unsigned long> ids)
        {
            ids_ = ids;
        }
        std::vector<unsigned long> getMapPointIds()
        {
            return ids_;
        }
    };

    struct superPoint_
    {
        superPoint_(){};
        superPoint_(kFrame_* kf, cv::Point2f pt, bool isout)
        {
            pt_ = pt;
            kf_ = kf;
            isOutLier_ = isout;
        }
        cv::Point2f pt_;
        kFrame_ *kf_;
        bool isOutLier_;
    };

    class mapPoint
    {
        private:
            unsigned long mid = 0;
            Eigen::Vector3d mpose;
            cv::Mat mdesc;
            std::vector<superPoint_*> mobservations_;
            int mobservationTimes = 0;

            std::mutex mMutex;
            
        public:
            mapPoint(cv::Point3f &mp);
            void addObservation(superPoint_* p);
            std::vector<superPoint_*> getObs();
            unsigned long getID();
            Eigen::Vector3d getPose();
            void setPose(Eigen::Vector3d p);
            ~mapPoint();
    };
    
}
#endif