#include <keyFrame.h>
#include <poseEstimation.h>
#include <opencv2/core/eigen.hpp>

namespace superVO
{
    keyFrame::keyFrame(){}

    keyFrame::keyFrame(std::shared_ptr<trtEngine> engine, std::shared_ptr<map> pmap, 
                    cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Mat extrinsic,
                    bool ifUndistort): mIfUndistort(ifUndistort)
    {
        mEngine = engine;
        mmap = pmap;
        mK = cameraMatrix.clone();
        mD = distCoeffs.clone();
        mExtrinsic = extrinsic.clone();
        mK.convertTo(mK, CV_32FC1);
        mD.convertTo(mD, CV_32FC1);
        mExtrinsic.convertTo(mExtrinsic, CV_32FC1);
    }

    void keyFrame::constractWorldFrame(std::string imageLeftFile, std::string imageRightFile, double timeStamp)
    {
        std::cout << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << "        Constract world key frame       " << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << imageLeftFile << std::endl;

        mImgFile = imageLeftFile;
        mTimeStamp = timeStamp;
        mKeyFrameImg = cv::imread(imageLeftFile, cv::IMREAD_COLOR);
        mRightImg = cv::imread(imageRightFile, cv::IMREAD_COLOR);
        // cv::resize(mKeyFrameImg, mKeyFrameImg, cv::Size(1240, 376), cv::INTER_NEAREST);
        // cv::resize(mRightImg, mRightImg, cv::Size(1240, 376), cv::INTER_NEAREST);
        
        undistort(mIfUndistort);

        auto t1 = std::chrono::steady_clock::now();
        std::vector<cv::Point3f> pts;
        mEngine->getPoints(mKeyFrameImg, pts, mdesc);
        auto t2 = std::chrono::steady_clock::now();
        auto timeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << mInfoType << "Extract points time used: " << timeUsed.count() << std::endl;
        for(auto &p: pts)
        {
            mkpts.emplace_back(cv::Point2i(p.x, p.y));
        }
        std::cout << mInfoType << "Extract pts: " << mkpts.size() << std::endl;
        
        findFeatureInRight();
        triangulation();
    }

    void keyFrame::constractWorldFrame(cv::Mat &imageLeft, cv::Mat &imageRight)
    {
        std::cout << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << "        Constract world key frame       " << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;

        // convert gray image to RGB
        if(imageLeft.channels() == 1)
        {
            mKeyFrameImg = gray2BGR(imageLeft);
            mRightImg = gray2BGR(imageRight);
            mDrawnImg = mKeyFrameImg.clone();
        }
        else
        {
            mKeyFrameImg = imageLeft.clone();
            mRightImg = mRightImg.clone();
            mDrawnImg = mKeyFrameImg.clone();
        }
        undistort(mIfUndistort);

        auto t1 = std::chrono::steady_clock::now();
        std::vector<cv::Point3f> pts;
        mEngine->getPoints(mKeyFrameImg, pts, mdesc);
        auto t2 = std::chrono::steady_clock::now();
        auto timeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << mInfoType << "Extract points time used: " << timeUsed.count() << std::endl;
        for(auto &p: pts)
        {
            mkpts.emplace_back(cv::Point2i(p.x, p.y));
        }
        std::cout << mInfoType << "Extract pts: " << mkpts.size() << std::endl;
        
        findFeatureInRight();
        triangulation();
    }

    void keyFrame::updateKeyFrame(std::string refFrameFile, std::string imageLeftFile, std::string imageRightFile, double timeStamp)
    {
        mSuccessSingle.clear();
        mError.clear();
        mRansacStatus.clear();
        mTriangulateSuccess.clear();
        mkpts3D.clear();
        mPose.clear();
        mkpts.clear();
        mkptsRight.clear();
        mrefPts.clear();
        mmapPoints.clear();

        std::cout << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << "           Get a new key frame          " << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << imageLeftFile << std::endl;

        mImgFile = imageLeftFile;
        mTimeStamp = timeStamp;
        mRefFrameImg = cv::imread(refFrameFile, cv::IMREAD_COLOR);
        mKeyFrameImg = cv::imread(imageLeftFile, cv::IMREAD_COLOR);
        mRightImg = cv::imread(imageRightFile, cv::IMREAD_COLOR);
        mDrawnImg = mKeyFrameImg.clone();
        
        undistort(mIfUndistort);
        
        auto t1 = std::chrono::steady_clock::now();
        std::vector<cv::Point3f> pts;
        mEngine->getPoints(mRefFrameImg, pts, mdesc);
        auto t2 = std::chrono::steady_clock::now();
        auto timeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << mInfoType << "Extract points time used: " << timeUsed.count() << std::endl;
        for(auto &p: pts)
        {
            mrefPts.emplace_back(cv::Point2i(p.x, p.y));
        }
        std::cout << mInfoType << "Extract pts: " << mrefPts.size() << std::endl;
        
        trackLastFrame();
        findFeatureInRight();
        triangulation();
    }

    void keyFrame::updateKeyFrame(cv::Mat &refFrame, cv::Mat &imageLeft, cv::Mat &imageRight)
    {
        mSuccessSingle.clear();
        mError.clear();
        mRansacStatus.clear();
        mTriangulateSuccess.clear();
        mkpts3D.clear();
        mPose.clear();
        mkpts.clear();
        mkptsRight.clear();
        mrefPts.clear();
        mmapPoints.clear();

        std::cout << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        std::cout << mInfoType << "           Get a new key frame          " << std::endl;
        std::cout << mInfoType << "----------------------------------------" << std::endl;
        
        // convert gray image to RGB
        if(imageLeft.channels() == 1)
        {
            mRefFrameImg = gray2BGR(refFrame);
            mKeyFrameImg = gray2BGR(imageLeft);
            mRightImg = gray2BGR(imageRight);
            mDrawnImg = mKeyFrameImg.clone();
        }
        else
        {
            mRefFrameImg = refFrame.clone();
            mKeyFrameImg = imageLeft.clone();
            mRightImg = imageRight.clone();
            mDrawnImg = mKeyFrameImg.clone();
        }

        auto t1 = std::chrono::steady_clock::now();
        std::vector<cv::Point3f> pts;
        mEngine->getPoints(mRefFrameImg, pts, mdesc);
        auto t2 = std::chrono::steady_clock::now();
        auto timeUsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
        std::cout << mInfoType << "Extract points time used: " << timeUsed.count() << std::endl;
        for(auto &p: pts)
        {
            mrefPts.emplace_back(cv::Point2i(p.x, p.y));
        }
        std::cout << mInfoType << "Extract pts: " << mrefPts.size() << std::endl;
        
        trackLastFrame();
        findFeatureInRight();
        triangulation();

    }

    void keyFrame::upDateMap()
    {
        std::vector<unsigned long> mapPointIds;
        for(int i = 0; i < mkpts3D.size(); i++)
        {
            mapPoint *newMapPoint = new mapPoint(mkpts3D[i]);
            mapPointIds.emplace_back(newMapPoint->getID());
            superPoint_ *newSuperPoint = new superPoint_(mkFrame, mkpts[i], true);
            newMapPoint->addObservation(newSuperPoint);
            mmap->insertMapPoint(newMapPoint);
            mmapPoints.emplace_back(newMapPoint);
        }
        mkFrame->setMapPointIds(mapPointIds);
    }


    void keyFrame::undistort(bool ifUndistort)
    {
        if(ifUndistort)
        {
            cv::Mat undistort, map1, map2;
            cv::initUndistortRectifyMap(mK, mD, cv::Mat(),
                                        cv::getOptimalNewCameraMatrix(mK, mD, cv::Size(mKeyFrameImg.cols, mKeyFrameImg.rows), 0, cv::Size(mKeyFrameImg.cols, mKeyFrameImg.rows), 0),
                                        cv::Size(mKeyFrameImg.cols, mKeyFrameImg.rows), CV_16SC2, map1, map2);

            cv::remap(mKeyFrameImg, undistort, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            mKeyFrameImg = undistort.clone();

            cv::remap(mRightImg, undistort, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            mRightImg = undistort.clone();

            cv::remap(mRefFrameImg, undistort, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            mRefFrameImg = undistort.clone();

            mDrawnImg = mKeyFrameImg.clone();
        }
        else
        {
            mDrawnImg = mKeyFrameImg.clone();
        }
    }

    /// @brief Change single channel image to three channels image
    /// @param src Image should be changed
    /// @return Three channels image
    cv::Mat keyFrame::gray2BGR(cv::Mat &src)
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

    /// @brief Start the LK optical flow tracking
    void keyFrame::trackLastFrame()
    {
        std::vector<cv::Point2f> kPts;

        cv::Mat keyFrameGray, refFrameGray;
        if(mRefFrameImg.channels() == 3)
        {
            cv::cvtColor(mRefFrameImg, refFrameGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(mKeyFrameImg, keyFrameGray, cv::COLOR_BGR2GRAY);
            cv::calcOpticalFlowPyrLK(refFrameGray, keyFrameGray, mrefPts, kPts, mSuccessSingle, mError);
        }
        else
        {
            cv::calcOpticalFlowPyrLK(mRefFrameImg, mKeyFrameImg, mrefPts, kPts, mSuccessSingle, mError);
        }

        cv::findFundamentalMat(mrefPts, kPts, mRansacStatus, cv::FM_RANSAC);

        for(int i = 0; i < kPts.size(); i++)
        {
            if(mSuccessSingle[i] && mRansacStatus[i])
            {
                mcountGood++;
                mkpts.emplace_back(cv::Point2i(kPts[i].x, kPts[i].y));
                cv::circle(mDrawnImg, cv::Point2i(kPts[i].x, kPts[i].y), 1, cv::Scalar(0, 255, 0), 2);
                // cv::circle(mDrawnImg, cv::Point2i(refPts[i].x, refPts[i].y), 1, cv::Scalar(0, 255, 0), 2);
            }
        }
        
        std::cout << mInfoType << "Keyframe tracked points: " << mkpts.size() << std::endl;
    }

    void keyFrame::findFeatureInRight()
    {
        std::vector<uchar> success;
        std::vector<float> error;
        cv::Mat leftImgGray, rightImgGray;
        if(mKeyFrameImg.channels() == 3)
        {
            cv::cvtColor(mKeyFrameImg, leftImgGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(mRightImg, rightImgGray, cv::COLOR_BGR2GRAY);
            cv::calcOpticalFlowPyrLK(leftImgGray, rightImgGray, mkpts, mkptsRight, success, error);
        }
        else
        {
            leftImgGray = mKeyFrameImg;
            rightImgGray = mRightImg;
            cv::calcOpticalFlowPyrLK(leftImgGray, rightImgGray, mkpts, mkptsRight, success, error);
        }

        cv::Mat fundamental;
        std::vector<uchar> ransacStatus;
        cv::findFundamentalMat(mkpts, mkptsRight, ransacStatus, cv::FM_RANSAC);

        std::vector<cv::Point2f> goodkpts;
        std::vector<cv::Point2f> goodkptsRight;
        for(int i = 0; i < mkpts.size(); i++)
        {
            if(success[i] && ransacStatus[i])
            {
                goodkpts.emplace_back(mkpts[i]);
                goodkptsRight.emplace_back(mkptsRight[i]);
                cv::circle(mDrawnImg, cv::Point2i(mkpts[i].x, mkpts[i].y), 1, cv::Scalar(0, 255, 0), 2);
                // cv::circle(mRightImg, cv::Point2i(mkptsRight[i].x, mkptsRight[i].y), 1, cv::Scalar(0, 255, 0), 2);
            }
        }
        mkpts.clear();
        mkptsRight.clear();
        mkpts = goodkpts;
        mkptsRight = goodkptsRight;
    }

    void keyFrame::filtNew3DPoints()
    {
        // filter out 3Dpoints that different from last keyFrame
        // using desc to match
        torch::Tensor dmat = torch::mm(mlastDesc.t(), mdesc);
        dmat = torch::sqrt(0.5 - 0.5 * torch::clip(dmat, -1, 1));
        torch::Tensor idx = torch::argmin(dmat, 1, false);
        torch::Tensor scores = dmat.index({torch::arange(dmat.sizes()[0]), idx});
        torch::Tensor keep = scores < 0.7;
        torch::Tensor idx2 = torch::argmin(dmat, 0, false);
        torch::Tensor keep_bi = torch::arange(idx.sizes()[0]) == idx2.index_select(0, idx);
        keep = torch::logical_and(keep, keep_bi);
        idx = idx.index({keep});
        scores = scores.index({keep});

        torch::Tensor m_idx1 = torch::arange(mlastDesc.sizes()[1]).index({keep});
        torch::Tensor m_idx2 = idx;

        std::cout << "m_idx1: " << m_idx1.sizes() << std::endl;
        std::cout << "m_idx2: " << m_idx2.sizes() << std::endl;
    }


    /// @brief Update the keyframe pose
    /// @param refFrameSE3 Reference pose of last frame
    /// @param isWorldKeyFrame If the keyframe is a world frame
    void keyFrame::updateKeyFramePose(Eigen::Isometry3d refFrameSE3)
    {
        poseEstimation3d2d(mkpts3D, mkpts, mRwk, mtwk, mK);
        mRwk.convertTo(mRwk, CV_32FC1);
        mtwk.convertTo(mtwk, CV_32FC1);
        mTwk = (cv::Mat_<float>(3, 4) << mRwk.at<float>(0, 0), mRwk.at<float>(0, 1), mRwk.at<float>(0, 2), mtwk.at<float>(0),
                                        mRwk.at<float>(1, 0), mRwk.at<float>(1, 1), mRwk.at<float>(1, 2), mtwk.at<float>(1),
                                        mRwk.at<float>(2, 0), mRwk.at<float>(2, 1), mRwk.at<float>(2, 2), mtwk.at<float>(2));

        Eigen::Matrix3d R_;
        Eigen::Vector3d t_;
        cv::cv2eigen(mRwk, R_);
        cv::cv2eigen(mtwk, t_);
        Eigen::Quaterniond q_relative(R_);
        q_relative.normalize();
        Eigen::Isometry3d se3(q_relative);
        se3.pretranslate(t_);
        mse3 = refFrameSE3 * se3.inverse();
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

        kFrame_ *tempFrame = new kFrame_(mse3);
        mmap->insertMapFrame(tempFrame);
        mkFrame = tempFrame;

    }

    std::vector<double> keyFrame::getPose()
    {
        return mPose;
    }

    torch::Tensor keyFrame::getDesc()
    {
        return mdesc;
    }

    cv::Mat keyFrame::getSourceImage()
    {
        return mKeyFrameImg;
    }

    cv::Mat keyFrame::getDrawnImage()
    {
        return mDrawnImg;
    }

    inline cv::Point2f keyFrame::pixel2cam(cv::Point2f &p, const cv::Mat &K) 
    {
        return cv::Point2f
        (
            (p.x - K.at<float>(0, 2)) / K.at<float>(0, 0),
            (p.y - K.at<float>(1, 2)) / K.at<float>(1, 1)
        );
    }
    inline cv::Point2i keyFrame::cam2pixel(cv::Point3f &p3d, const cv::Mat &K)
    {
        return cv::Point2i
        (
            p3d.x / p3d.z * K.at<float>(0, 0) + K.at<float>(0, 2),
            p3d.y / p3d.z * K.at<float>(1, 1) + K.at<float>(1, 2)
        );
    }

    inline void keyFrame::Triangulate(const cv::Point2f &kp1, const cv::Point2f &kp2, cv::Mat &P1, cv::Mat &P2, cv::Mat &x3D)
    {
        cv::Mat A(4,4,CV_32F);
        
        A.row(0) = kp1.x * P1.row(2) - P1.row(0);
        A.row(1) = kp1.y * P1.row(2) - P1.row(1);
        A.row(2) = kp2.x * P2.row(2) - P2.row(0);
        A.row(3) = kp2.y * P2.row(2) - P2.row(1);
        cv::Mat u,w,vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
        x3D = vt.row(3).t();
        x3D = x3D.rowRange(0,3) / x3D.at<float>(3);//  转换成非齐次坐标  归一化

    }


    void keyFrame::triangulation()
    {
        std::vector<cv::Point2f> goodPts;

        cv::Mat T0 = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0,
                                               0, 1, 0, 0,
                                               0, 0, 1, 0);
        cv::Mat extrinsic_R = mExtrinsic.rowRange(0, 3).colRange(0, 3);
        cv::Mat extrinsic_t = mExtrinsic.rowRange(0, 3).col(3);
        for(int i = 0; i < mkpts.size(); i++)
        {
            cv::Mat x;
            Triangulate(pixel2cam(mkpts[i], mK), pixel2cam(mkptsRight[i], mK), T0, mExtrinsic, x);
            if(x.at<float>(2) < 0 || x.at<float>(2) > 30)
            {
                mTriangulateSuccess.emplace_back(false);
                continue;
            }
            mTriangulateSuccess.emplace_back(true);

            cv::Mat xp = (cv::Mat_<float>(3, 1) << x.at<float>(0), x.at<float>(1), x.at<float>(2));
            cv::Point3f p3dRefLeft(xp.at<float>(0, 0), xp.at<float>(1, 0), xp.at<float>(2, 0));
            xp = extrinsic_R * xp + extrinsic_t;
            cv::Point3f p3dRefRight(xp.at<float>(0, 0), xp.at<float>(1, 0), xp.at<float>(2, 0));
            cv::Point2i reprojectPoint = cam2pixel(p3dRefLeft, mK);
            
            // compute reproject error
            float reprojectErr = std::sqrt((mkpts[i].x - reprojectPoint.x) * (mkpts[i].x - reprojectPoint.x) + 
                                         (mkpts[i].y - reprojectPoint.y) * (mkpts[i].y - reprojectPoint.y));
            if(reprojectErr > mreprojectErrorTh)
            {
                mTriangulateSuccess.emplace_back(false);
                continue;
            }
            mkpts3D.emplace_back(p3dRefLeft);
            cv::Point2i reprojectPointRight = cam2pixel(p3dRefRight, mK);
            cv::circle(mDrawnImg, reprojectPoint, 1, cv::Scalar(0, 0, 255), 2);

            goodPts.emplace_back(mkpts[i]);
        }
        mkpts.clear();
        mkpts = goodPts;
        std::cout << mInfoType << "Triangulated success: " << mkpts3D.size() << std::endl;
    }

    void keyFrame::setPts(std::vector<cv::Point2f> pts2d, std::vector<cv::Point3f> pts3d)
    {
        mkpts = pts2d;
        mkpts3D = pts3d;
    }

    void keyFrame::setPts(std::vector<cv::Point3f> pts3d)
    {
        mkpts3D = pts3d;
    }

    std::vector<mapPoint*> keyFrame::getPts3D()
    {
        return mmapPoints;
    }


    void keyFrame::setPose(cv::Mat T)
    {
        mTwk = T.clone();
        mRwk = T.rowRange(0, 3).colRange(0, 3);
        mtwk = T.rowRange(0, 3).col(3);
        Eigen::Matrix3d R_;
        Eigen::Vector3d t_;
        cv::cv2eigen(mRwk, R_);
        cv::cv2eigen(mtwk, t_);
        Eigen::Quaterniond q_relative(R_);
        q_relative.normalize();
        Eigen::Isometry3d se3(q_relative);
        se3.pretranslate(t_);
        mse3 = se3;

        kFrame_ *tempFrame = new kFrame_(mse3);
        mmap->insertMapFrame(tempFrame);
        mkFrame = tempFrame;
    }

    Eigen::Isometry3d keyFrame::getSE3()
    {
        return mse3;
    }

    std::string keyFrame::getImgFilePath()
    {
        return mImgFile;
    }

    kFrame_* keyFrame::getkeyFrame()
    {
        return mkFrame;
    }

    keyFrame::~keyFrame()
    {
        std::cout << "keyframe: " << mImgFile << " delete!" << std::endl;
    }
}
