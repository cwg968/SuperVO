#include <dataLoader.h>
#include <fstream>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <boost/format.hpp>


namespace superVO
{
    dataLoader::dataLoader(std::string settingPath)
    {
        cv::FileStorage settings(settingPath, cv::FileStorage::READ);
        settings["cameraMatrix"] >> mK;
        settings["distCoeffs"] >> mD;
        settings["extrinsic"] >> mextrinsic;
        settings["imageHeight"] >> mINPUT_H;
        settings["imageWidth"] >> mINPUT_W;
        settings["LKThresh"] >> mLKTHRESH;
        settings["ReprojectThresh"] >> mReprojectThresh;
        settings.release();
    }

    dataLoader::dataLoader(std::string settingPath, std::string dataPath)
    {
        cv::FileStorage settings(settingPath, cv::FileStorage::READ);
        settings["cameraMatrix"] >> mK;
        settings["distCoeffs"] >> mD;
        settings["extrinsic"] >> mextrinsic;
        settings["imageHeight"] >> mINPUT_H;
        settings["imageWidth"] >> mINPUT_W;
        settings["LKThresh"] >> mLKTHRESH;
        settings["ReprojectThresh"] >> mReprojectThresh;
        settings.release();
        std::cout << mInfoType << "Camera Intrinsic: " << std::endl << mK << std::endl;
        std::cout << mInfoType << "Camera Entrinsic: " << std::endl << mextrinsic << std::endl;

        // loadExtrinsic(dataPath);
        loadImages(dataPath);
    }

    cv::Mat dataLoader::K()
    {
        return mK;
    }

    cv::Mat dataLoader::D()
    {
        return mD;
    }

    cv::Mat dataLoader::extrisic()
    {
        return mextrinsic;
    }

    int dataLoader::INPUT_H()
    {
        return mINPUT_H;
    }

    int dataLoader::INPUT_W()
    {
        return mINPUT_W;
    }

    float dataLoader::LKTHRESH()
    {
        return mLKTHRESH;
    }

    float dataLoader::ReprojectThresh()
    {
        return mReprojectThresh;
    }

    std::string dataLoader::lastFrame()
    {
        return mimageLeft[mlastFrameId];
    }

    std::string dataLoader::lastFrameRight()
    {
        return mimageRight[mlastFrameId];
    }

    std::string dataLoader::nextLeftFrame()
    {
        mlastFrameId = mcurrentFrameId;
        return mimageLeft[mcurrentFrameId++];
    }

    std::string dataLoader::nextRightFrame()
    {
        return mimageRight[mcurrentFrameId];
    }

    double dataLoader::getCurFrameTime()
    {
        return mtimeStamps[mcurrentFrameId];
    }

    double dataLoader::getLastFrameTime()
    {
        return mtimeStamps[mlastFrameId];
    }

    void dataLoader::loadExtrinsic(std::string path)
    {
        std::ifstream fin(path + "calib.txt");
        if (!fin)
        {
            std::cout << mInfoType << "cannot find " << path << "/calib.txt!";
            exit(1);
        }

        for (int i = 0; i < 2; ++i)
        {
            char camera_name[3];
            for (int k = 0; k < 3; ++k)
            {
                fin >> camera_name[k];
            }
            double projection_data[12];
            for (int k = 0; k < 12; ++k)
            {
                fin >> projection_data[k];
            }
            Eigen::Matrix3d K;
            K << projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10];
            Eigen::Vector3d t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            Sophus::SE3d ex(Sophus::SO3d(), t);
            if(i == 1)
            {
                mextrinsic = (cv::Mat_<float>(3, 4) << ex.matrix()(0, 0), ex.matrix()(0, 1), ex.matrix()(0, 2), ex.matrix()(0, 3),
                                                    ex.matrix()(1, 0), ex.matrix()(1, 1), ex.matrix()(1, 2), ex.matrix()(1, 3),
                                                    ex.matrix()(2, 0), ex.matrix()(2, 1), ex.matrix()(2, 2), ex.matrix()(2, 3));
            }
        }
        fin.close();
        std::cout << mInfoType << "extrinsic: " << mextrinsic << std::endl;
    }

    void dataLoader::loadImages(std::string path)
    {
        std::ifstream fTime;
        std::ifstream fImage;
        std::string timeStampPath = path + "times.txt";
        std::string imagePath = path + "rgb.txt";
        std::cout << mInfoType << "path: " << path << std::endl;
        fTime.open(timeStampPath.c_str());
        fImage.open(imagePath.c_str());
        if(fTime.is_open())
        {
            std::cout << mInfoType << "times.txt loaded success!" << std::endl;
        }
        else
        {
            std::cout << mInfoType << "times.txt loaded failed!" << std::endl;
            exit(1);
        }

        if(fImage.is_open())
        {
            std::cout << mInfoType << "rgb.txt loaded success!" << std::endl;
        }
        else
        {
            std::cout << mInfoType << "rgb.txt loaded failed!" << std::endl;
            exit(1);
        }
        

        while(!fTime.eof())
        {
            std::string sTime;
            std::string sImage;
            std::getline(fTime,sTime);
            std::getline(fImage, sImage);
            if(!sTime.empty())
            {
                std::stringstream ss;
                ss << sTime;
                double t;
                ss >> t;
                mtimeStamps.emplace_back(t);
            }
            if(!sImage.empty())
            {
                std::stringstream ss;
                ss << sImage;
                std::string imgLeft;
                ss >> imgLeft;
                mimageLeft.emplace_back(path + "/image_0/" + imgLeft);
                mimageRight.emplace_back(path + "/image_1/" + imgLeft);
            }
        }
    }

    dataLoader::~dataLoader()
    {
    }

}