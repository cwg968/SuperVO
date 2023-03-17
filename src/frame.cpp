#include <frame.h>
#include <poseEstimation.h>
#include <opencv2/core/eigen.hpp>
#include <sys/time.h>

namespace superVO
{
    frame::frame(){}

    frame::frame(std::shared_ptr<keyFrame> keyFrame_, 
                cv::Mat cameraMatrix, cv::Mat distCoeffs, 
                bool ifUndistort): mKeyFrame(keyFrame_)
    {   
        mIfUndistort = ifUndistort;
        mK = cameraMatrix.clone();
        mD = distCoeffs.clone();
        mK.convertTo(mK, CV_32FC1);
        mD.convertTo(mD, CV_32FC1);
    }

    void frame::updateFrame(std::string frameFile, double timeStamp)
    {
        mSuccessSingle.clear();
        mError.clear();
        mRansacStatus.clear();
        mPts3D.clear();
        mPose.clear();
        mfpts.clear();
        mkFrame = mKeyFrame->getkeyFrame();

        std::cout << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << "             Track a frame              " << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << "Keyframe: " << mKeyFrame->getImgFilePath() << std::endl;
        std::cout << mInfoType << frameFile << std::endl;

        mImgFile = frameFile;
        mTimeStamp = timeStamp;

        undistort(mIfUndistort);
        LKTrack();
        updatePoints();
    }

    void frame::updateFrame(cv::Mat &frameFile)
    {
        mSuccessSingle.clear();
        mError.clear();
        mRansacStatus.clear();
        mPts3D.clear();
        mPose.clear();
        mfpts.clear();

        std::cout << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << "             Track a frame              " << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;

        if(frameFile.channels() == 1)
        {
            mFrameImg = gray2BGR(frameFile);
            mDrawnImg = mFrameImg.clone();
        }
        else
        {
            mFrameImg = frameFile.clone();
            mDrawnImg = mFrameImg.clone();
        }
        LKTrack();
        updatePoints();

        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        mTimeStamp = now.count() / 1000.0;
    }

    void frame::undistort(bool ifUndistort)
    {
        mFrameImg = cv::imread(mImgFile, cv::IMREAD_COLOR);
        // cv::resize(mFrameImg, mFrameImg, cv::Size(1240, 376), cv::INTER_NEAREST);

        if(ifUndistort)
        {
            cv::Mat undistort, map1, map2;
            // cv::undistort(mFrameImg, undistort, K, D, K);
            cv::initUndistortRectifyMap(mK, mD, cv::Mat(),
                                        cv::getOptimalNewCameraMatrix(mK, mD, cv::Size(mFrameImg.cols, mFrameImg.rows), 0, cv::Size(mFrameImg.cols, mFrameImg.rows), 0),
                                        cv::Size(mFrameImg.cols, mFrameImg.rows), CV_16SC2, map1, map2);
            cv::remap(mFrameImg, undistort, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            mFrameImg = undistort.clone();
            mDrawnImg = mFrameImg.clone();
        }
        else
        {
            mDrawnImg = mFrameImg.clone();
        }
        
    }

    void frame::LKTrack()
    {
        cv::Mat frameImgGray, keyFrameImgGray;
        cv::cvtColor(mFrameImg, frameImgGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(mKeyFrame->mKeyFrameImg, keyFrameImgGray, cv::COLOR_BGR2GRAY);

        cv::calcOpticalFlowPyrLK(keyFrameImgGray, frameImgGray, mKeyFrame->mkpts, mfpts, mSuccessSingle, mError);
        cv::findFundamentalMat(mKeyFrame->mkpts, mfpts, mRansacStatus, cv::FM_RANSAC);
    }

    cv::Mat frame::gray2BGR(cv::Mat &src)
    {
        cv::Mat imageColor = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
        std::vector<cv::Mat> channels;
        for (int i=0;i<3;i++)
        {
            channels.push_back(src);
        }
        cv::merge(channels, imageColor);
        return imageColor;
    }

    void frame::updateFramePose()
    {
        poseEstimation3d2d(mPts3D, mfpts, mRkf, mtkf, mK);
        mRkf.convertTo(mRkf, CV_32FC1);
        mtkf.convertTo(mtkf, CV_32FC1);
        mTkf = (cv::Mat_<float>(3, 4) << mRkf.at<float>(0, 0), mRkf.at<float>(0, 1), mRkf.at<float>(0, 2), mtkf.at<float>(0),
                                        mRkf.at<float>(1, 0), mRkf.at<float>(1, 1), mRkf.at<float>(1, 2), mtkf.at<float>(1),
                                        mRkf.at<float>(2, 0), mRkf.at<float>(2, 1), mRkf.at<float>(2, 2), mtkf.at<float>(2));

        Eigen::Matrix3d R_;
        Eigen::Vector3d t_;
        cv::cv2eigen(mRkf, R_);
        cv::cv2eigen(mtkf, t_);

        Eigen::Quaterniond q_relative(R_);
        q_relative.normalize();
        Eigen::Isometry3d se3(q_relative);
        se3.pretranslate(t_);
        mse3 = mKeyFrame->mse3 * se3; 
        R_ << mse3.matrix()(0, 0), mse3.matrix()(0, 1), mse3.matrix()(0, 2),
              mse3.matrix()(1, 0), mse3.matrix()(1, 1), mse3.matrix()(1, 2),
              mse3.matrix()(2, 0), mse3.matrix()(2, 1), mse3.matrix()(2, 2);
        t_ << mse3.matrix()(0, 3), mse3.matrix()(1, 3), mse3.matrix()(2, 3);

        Eigen::Quaterniond q_absolute(R_);
        q_absolute.normalize();

        mPose.emplace_back(mTimeStamp);
        mPose.emplace_back(t_(0));
        mPose.emplace_back(t_(1));
        mPose.emplace_back(t_(2));
        mPose.emplace_back(q_absolute.coeffs()(0));
        mPose.emplace_back(q_absolute.coeffs()(1));
        mPose.emplace_back(q_absolute.coeffs()(2));
        mPose.emplace_back(q_absolute.coeffs()(3));
    }

    void frame::updatePoints()
    {
        std::vector<mapPoint*> refPts3D = mKeyFrame->getPts3D();
        // std::vector<cv::Point3f> refPts3D = mKeyFrame->getPts3D();
        std::vector<cv::Point2f> goodfpts;
        // draw keypoints
        auto t1 = refPts3D.size();
        auto t2 = mfpts.size();
        for(int i = 0; i < mfpts.size(); i++)
        {
            if(mSuccessSingle[i] && mRansacStatus[i])
            {
                mcountGood++;
                goodfpts.emplace_back(mfpts[i]);
                auto mapPose = refPts3D[i]->getPose();
                mPts3D.emplace_back(cv::Point3f(mapPose.x(), mapPose.y(), mapPose.z()));

                superPoint_ *newSuperPOint = new superPoint_(mkFrame, mfpts[i], true);
                refPts3D[i]->addObservation(newSuperPOint);
                
                cv::circle(mDrawnImg, cv::Point2i(mfpts[i].x, mfpts[i].y), 1, cv::Scalar(0, 255, 0), 2);
                // cv::circle(mDrawnImg, cam2pixel(refPts3D[i], mK), 1, cv::Scalar(255, 0, 0), 2);
                // cv::line(mDrawnImg, cam2pixel(refPts3D[i], mK), cv::Point2i(mfpts[i].x, mfpts[i].y), cv::Scalar(0, 255, 0), 1);
            }
        }
        mfpts = goodfpts;

        std::cout << mInfoType << "Tracked pts: " << mfpts.size();
    }

    float frame::reproject()
    {
        std::vector<cv::Point3f> newPts3D;
        int reprojectError = 0;
        for(int i = 0; i < mPts3D.size(); i++)
        {
            cv::Mat x = (cv::Mat_<float>(3, 1) << mPts3D[i].x, mPts3D[i].y, mPts3D[i].z);
            mRkf.convertTo(mRkf, CV_32FC1);
            mtkf.convertTo(mtkf, CV_32FC1);
            x = mRkf * x + mtkf;
            cv::Point3f p3d(x.at<float>(0), x.at<float>(1), x.at<float>(2));
            cv::Point2i p3dRepro = cam2pixel(p3d, mK);
            int squareError = (p3dRepro.x - mfpts[i].x) * (p3dRepro.x - mfpts[i].x) + 
                            (p3dRepro.y - mfpts[i].y) * (p3dRepro.y - mfpts[i].y);

            if(squareError < mReprojectThresh)
            {
                reprojectError += squareError;
                newPts3D.emplace_back(mPts3D[i]);
                cv::circle(mDrawnImg, p3dRepro, 1, cv::Scalar(0, 0, 255), 2);
            }
        }
        mPts3D = newPts3D;
        std::cout << mInfoType << "Reprojected success pts3D: " << mPts3D.size();
        return reprojectError;
    }

    std::vector<double> frame::getPose()
    {
        return mPose;
    }

    cv::Mat frame::getT()
    {
        return mTkf;
    }

    int frame::getGoodPtsNum()
    {
        return mcountGood;
    }

    int frame::getPtsNum()
    {
        return mfpts.size();
    }

    std::vector<cv::Point3f> frame::getPts3D()
    {
        return mPts3D;
    }

    Eigen::Isometry3d frame::getSE3()
    {
        return mse3;
    }

    cv::Mat frame::getSourceImage()
    {
        return mFrameImg;
    }

    cv::Mat frame::getDrawnImage()
    {
        return mDrawnImg;
    }

    inline cv::Point2f frame::pixel2cam(cv::Point2f &p, const cv::Mat &K) 
    {
    return cv::Point2f
        (
            (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
            (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
        );
    }
    inline cv::Point2i frame::cam2pixel(cv::Point3f &p3d, const cv::Mat &K)
    {
        return cv::Point2i
        (
            p3d.x / p3d.z * K.at<float>(0, 0) + K.at<float>(0, 2),
            p3d.y / p3d.z * K.at<float>(1, 1) + K.at<float>(1, 2)
        );
    }

    std::string frame::getImgFilePath()
    {
        return mImgFile;
    }

    frame::~frame()
    {
        std::cout << "frame: " << mImgFile << " delete!" << std::endl;
    }
}



