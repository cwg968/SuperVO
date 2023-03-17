#include <dataLoader.h>
#include <fstream>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <boost/format.hpp>


namespace superVO
{
    dataLoader::dataLoader(std::string dataPath, std::string settingPath)
    {
        cv::FileStorage settings(settingPath, cv::FileStorage::READ);
        settings["cameraMatrix"] >> mK;
        settings["distCoeffs"] >> mD;
        settings["imageHeight"] >> mINPUT_H;
        settings["imageWidth"] >> mINPUT_W;
        settings["LKThresh"] >> mLKTHRESH;
        settings["ReprojectThresh"] >> mReprojectThresh;
        settings.release();

        loadExtrinsic(dataPath);
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
        std::ifstream f;
        std::string timeStampPath = path + "times.txt";
        std::cout << mInfoType << "path: " << path << std::endl;
        f.open(timeStampPath.c_str());
        if(f.is_open())
        {
            std::cout << mInfoType << "times.txt loaded success!" << std::endl;
        }
        else
        {
            std::cout << mInfoType << "times.txt loaded failed!" << std::endl;
            exit(1);
        }
        
        boost::format fmt("%s/image_%d/%06d.png");
        int imageNum = 0;

        while(!f.eof())
        {
            std::string s;
            std::getline(f,s);
            if(!s.empty())
            {
                std::stringstream ss;
                ss << s;
                double t;
                std::string imgLeft, imgRight;
                ss >> t;
                mtimeStamps.emplace_back(t);
                mimageLeft.emplace_back((fmt % path % 0 % imageNum).str());
                mimageRight.emplace_back((fmt % path % 1 % imageNum).str());
                imageNum++;
            }
        }
    }

    dataLoader::~dataLoader()
    {
    }

}