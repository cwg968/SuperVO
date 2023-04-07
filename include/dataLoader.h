#ifndef DATALOADER_H
#define DATALOADER_H

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

namespace superVO
{
    class dataLoader
    {
        private:
            cv::Mat mK;
            cv::Mat mD;
            cv::Mat mextrinsic;
            int mINPUT_H;
            int mINPUT_W;
            float mLKTHRESH;
            float mReprojectThresh;

            std::vector<std::string> mimageLeft;
            std::vector<std::string> mimageRight;
            std::vector<double> mtimeStamps;

            int mcurrentFrameId = 0;
            int mlastFrameId = 0;

            std::string mInfoType = "[DATALOADER]-->";

        public:
            dataLoader(std::string settingPath);
            dataLoader(std::string settingPath, std::string dataPath);
            cv::Mat K();
            cv::Mat D();
            cv::Mat extrisic();
            int INPUT_H();
            int INPUT_W();
            float LKTHRESH();
            float ReprojectThresh();
            std::string lastFrame();
            std::string lastFrameRight();
            std::string nextLeftFrame();
            std::string nextRightFrame();
            double getCurFrameTime();
            double getLastFrameTime();
            ~dataLoader();

        private:
            void loadExtrinsic(std::string path);
            void loadImages(std::string path);
    };
}




#endif