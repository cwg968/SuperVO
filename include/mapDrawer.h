#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <mutex>

namespace superVO
{
    class mapDrawer
    {
        private:
            float mKeyFrameSize;
            float mKeyFrameLineWidth;
            float mGraphLineWidth;
            float mPointSize;
            float mCameraSize;
            float mCameraLineWidth;

            float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;

            pangolin::OpenGlMatrix Twc;
            std::mutex mposeMutex;
        private:
            pangolin::OpenGlMatrix getOpenGLCameraMatrix(cv::Mat m);

        public:
            cv::Mat mCameraPose = cv::Mat::eye(4, 4, CV_64F);
        public:
            mapDrawer();
            void run();
            void drawKeyFrame(cv::Mat pose);
            void drawFrame(pangolin::OpenGlMatrix T);
            ~mapDrawer();
    };
}

#endif