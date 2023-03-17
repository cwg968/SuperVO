#ifndef FRAME_H
#define FRAME_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <keyFrame.h>

namespace superVO
{
    class frame
    {
        private:
            std::shared_ptr<keyFrame> mKeyFrame;
            std::vector<uchar> mSuccessSingle;
            std::vector<float> mError;
            std::vector<uchar> mRansacStatus;
            Eigen::Quaterniond mq; 
            std::vector<cv::Point3f> mPts3D;
            std::vector<double> mPose;

            cv::Mat mRkf; // The pose of this frame relative to the keyframe
            cv::Mat mtkf; 
            cv::Mat mTkf; //（mRkf|mtkf）
            cv::Mat mFrameImg;
            cv::Mat mDrawnImg;
            Eigen::Isometry3d mse3;
            std::string mImgFile;
            std::vector<cv::Point2f> mfpts;
            int mcountGood = 0;

            float mReprojectThresh = 10;

            std::string mInfoType = "[FRAME]-->";

            cv::Mat mK;
            cv::Mat mD;
            cv::Mat mNormal;

            double mTimeStamp;

            bool mIfUndistort;

            kFrame_* mkFrame;

        private:
            void updatePoints();
            void LKTrack();
            cv::Mat gray2BGR(cv::Mat &src);
            inline cv::Point2i cam2pixel(cv::Point3f &p3d, const cv::Mat &K);
            inline cv::Point2f pixel2cam(cv::Point2f &p, const cv::Mat &K);

        public:
            frame();

            frame(std::shared_ptr<keyFrame> keyFrame_, 
                cv::Mat cameraMatrix, cv::Mat distCoeffs, bool ifUndistort);

            void updateFrame(std::string frameFile, double timeStamp);
            void updateFrame(cv::Mat &frameFile);

            void undistort(bool ifUndistort);
            void updateFramePose();
            float reproject();
            int getGoodPtsNum();
            int getPtsNum();
            std::vector<cv::Point3f> getPts3D();
            Eigen::Isometry3d getSE3();
            cv::Mat getSourceImage();
            cv::Mat getDrawnImage();
            std::vector<double> getPose();
            cv::Mat getT();
            std::string getImgFilePath();
            ~frame();
    };
}

#endif