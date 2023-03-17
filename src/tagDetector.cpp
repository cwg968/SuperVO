#include <tagDetector.h>

tagDetector::tagDetector(const cv::Mat K, const cv::Mat D)
{
    mtf = tag36h11_create();
    mtd = apriltag_detector_create();

    apriltag_detector_add_family(mtd, mtf);
    mtd->quad_decimate = 1.0;
    mtd->quad_sigma    = 0.0;
    mtd->nthreads      = 1;
    mtd->debug         = 0;
    mtd->refine_edges  = 1;
    
    minfo.tagsize = 0.146 - 0.012 * 4;
    minfo.det = NULL;
    minfo.fx = K.at<float>(0, 0);
    minfo.fy = K.at<float>(1, 1);
    minfo.cx = K.at<float>(0, 2);
    minfo.cy = K.at<float>(1, 2);
}

bool tagDetector::detect(cv::Mat &frameImg)
{
    errno = 0;
    mDrawnFrameImg = cv::Mat::zeros(frameImg.rows, frameImg.cols, CV_8UC3);
    std::vector<cv::Mat> channels;
    for (int i=0;i<3;i++)
    {
        channels.push_back(frameImg);
    }
    cv::merge(channels,mDrawnFrameImg);

    image_u8_t im = {.width = frameImg.cols, .height = frameImg.rows, .stride = frameImg.cols, .buf = frameImg.data};
    zarray_t *detections = apriltag_detector_detect(mtd, &im);

    if(errno == EAGAIN)
    {   
        printf("Unable to create the %d threads requested.\n",mtd->nthreads);
        exit(-1);
    }
    for(int i = 0; i < zarray_size(detections); i++)
    {
        apriltag_detection_t *det;
        zarray_get(detections, i, &det);
        minfo.det = det;
        apriltag_pose_t pose;
        estimate_tag_pose(&minfo, &pose);
        cv::Mat R = (cv::Mat_<float>(3, 3) << pose.R->data[0], pose.R->data[1], pose.R->data[2], 
                                                pose.R->data[3], pose.R->data[4], pose.R->data[5], 
                                                pose.R->data[6], pose.R->data[7], pose.R->data[8]);
        cv::Mat t = (cv::Mat_<float>(3, 1) << pose.t->data[0], pose.t->data[1], pose.t->data[2]);
        Eigen::Matrix3d R_;
        Eigen::Vector3d t_;
        cv::cv2eigen(R, R_);
        cv::cv2eigen(t, t_);
        Eigen::Quaterniond q(R_);

        Eigen::Isometry3d se3(q);
        mse3 = se3;
        mse3.pretranslate(t_);

        std::cout << mInfoType << "-----------camera-----------" << std::endl;
        std::cout << mInfoType << "R: " << R_ << std::endl;
        std::cout << mInfoType << "t: " << t_ << std::endl;
        std::cout << mInfoType << "mse3: " << mse3.matrix() << std::endl;

        cv::line(mDrawnFrameImg, cv::Point(det->p[0][0], det->p[0][1]), cv::Point(det->p[1][0], det->p[1][1]), cv::Scalar(0, 255, 0), 1);
        cv::line(mDrawnFrameImg, cv::Point(det->p[0][0], det->p[0][1]), cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 255, 0), 1);
        cv::line(mDrawnFrameImg, cv::Point(det->p[1][0], det->p[1][1]), cv::Point(det->p[2][0], det->p[2][1]), cv::Scalar(0, 255, 0), 1);
        cv::line(mDrawnFrameImg, cv::Point(det->p[2][0], det->p[2][1]), cv::Point(det->p[3][0], det->p[3][1]), cv::Scalar(0, 255, 0), 1);
        cv::rectangle(mDrawnFrameImg, cv::Point(det->p[0][0] - 3, det->p[0][1] - 3), cv::Point(det->p[0][0] + 3, det->p[0][1] + 3), cv::Scalar(0, 255, 0), 1);
        cv::rectangle(mDrawnFrameImg, cv::Point(det->p[1][0] - 3, det->p[1][1] - 3), cv::Point(det->p[1][0] + 3, det->p[1][1] + 3), cv::Scalar(0, 255, 0), 1);
        cv::rectangle(mDrawnFrameImg, cv::Point(det->p[2][0] - 3, det->p[2][1] - 3), cv::Point(det->p[2][0] + 3, det->p[2][1] + 3), cv::Scalar(0, 255, 0), 1);
        cv::rectangle(mDrawnFrameImg, cv::Point(det->p[3][0] - 3, det->p[3][1] - 3), cv::Point(det->p[3][0] + 3, det->p[3][1] + 3), cv::Scalar(0, 255, 0), 1);

        std::stringstream ss;
        ss << det->id;
        std::string text = ss.str();
        int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        float fontscale = 1.0;
        int baseline;
        cv::Size textsize = cv::getTextSize(text, fontface, fontscale, 2, &baseline);
        cv::putText(mDrawnFrameImg, text, cv::Point(det->c[0]-textsize.width/2, det->c[1]+textsize.height/2), fontface, fontscale, cv::Scalar(0, 255, 0), 2);
        
        return true;
    }
    return false;
}

tagDetector::~tagDetector()
{
    apriltag_detector_destroy(mtd);
    tag36h11_destroy(mtf);
}


