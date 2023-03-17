#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <trtEngine.h>
#include <mapPoint.h>
#include <map.h>
#include <torch/torch.h>

namespace superVO
{
    class keyFrame
    {
        private:
            Eigen::Quaterniond mq;
            std::vector<uchar> mSuccessSingle;
            std::vector<float> mError;
            std::vector<uchar> mRansacStatus;
            std::vector<bool> mTriangulateSuccess;
            std::vector<cv::Point3f> mkpts3D;
            std::vector<mapPoint*> mmapPoints;
            std::vector<double> mPose;
            std::string mInfoType = "[KEYFRAME]-->";
            bool mIfUndistort;
            cv::Mat mK;
            cv::Mat mD;
            cv::Mat mExtrinsic;
            std::string mImgFile;

            torch::Tensor mdesc;
            torch::Tensor mlastDesc;

            double mTimeStamp;

            std::shared_ptr<trtEngine> mEngine;

            kFrame_* mkFrame;
            std::shared_ptr<map> mmap;

            int mreprojectErrorTh = 5;

        private:
            cv::Mat gray2BGR(cv::Mat &src);
            inline cv::Point2i cam2pixel(cv::Point3f &p3d, const cv::Mat &K);
            inline cv::Point2f pixel2cam(cv::Point2f &p, const cv::Mat &K);
            inline void Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, cv::Mat &P1, cv::Mat &P2, cv::Mat &x3D);
            void triangulation();
            void undistort(bool ifUndistort);
            void trackLastFrame();
            void findFeatureInRight();
            void filtNew3DPoints();

        public:
            cv::Mat mRwk;
            cv::Mat mtwk; // The pose of this frame relative to the keyframe
            cv::Mat mTwk; // (mRwk|mtwk)
            std::vector<cv::Point2f> mkpts;
            std::vector<cv::Point2f> mkptsRight;
            std::vector<cv::Point2f> mrefPts;
            Eigen::Isometry3d mse3;
            cv::Mat mKeyFrameImg;
            cv::Mat mRightImg;
            cv::Mat mDrawnImg;
            cv::Mat mRefFrameImg;
            int mcountGood = 0;

        public:
            keyFrame();

            keyFrame(std::shared_ptr<trtEngine> engine, std::shared_ptr<map> pmap, 
                    cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Mat extrinsic,
                    bool ifUndistort);

            void constractWorldFrame(std::string imageLeftFile, std::string imageRightFile, double timeStamp);
            void constractWorldFrame(cv::Mat &imageLeft, cv::Mat &imageRight);
            void updateKeyFrame(std::string refFrameFile, std::string imageLeftFile, std::string imageRightFile, double timeStamp);
            void updateKeyFrame(cv::Mat &refFrame, cv::Mat &imageLeft, cv::Mat &imageRight);

            void upDateMap();

            std::vector<double> getPose();
            torch::Tensor getDesc();
            void updateKeyFramePose(Eigen::Isometry3d keyFrameSE3);
            cv::Mat getSourceImage();
            cv::Mat getDrawnImage();
            void setPts(std::vector<cv::Point2f> pts2d, std::vector<cv::Point3f> pts3d);
            void setPts(std::vector<cv::Point3f> pts3d);
            std::vector<mapPoint*> getPts3D();
            void setPose(cv::Mat T);
            Eigen::Isometry3d getSE3();
            std::string getImgFilePath();
            kFrame_* getkeyFrame();
            ~keyFrame();
    };
}


#endif
