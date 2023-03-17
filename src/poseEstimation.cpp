#include <poseEstimation.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <fstream>

using namespace cv;
using namespace std;

void poseEstimation2d2d(std::vector<cv::Point2f> kpts1, std::vector<cv::Point2f> pts2, cv::Mat &R, cv::Mat &t, cv::Mat &cameraMatrix)
{
    cv::Point2f principal_point(318.64304, 255.313989);
    float focal_length = 517.306408;
    cv::Mat E = cv::findEssentialMat(kpts1, pts2, cameraMatrix);
    
    cv::recoverPose(E, kpts1, pts2, cameraMatrix, R, t);
}

void poseEstimation3d2d(std::vector<cv::Point3f> p3d, std::vector<cv::Point2f> p2d, cv::Mat &R, cv::Mat &t, cv::Mat &cameraMatrix)
{
    cv::Mat r;
    cv::solvePnP(p3d, p2d, cameraMatrix, cv::Mat(), r, t, false);
    cv::Rodrigues(r, R);
}

void poseEstimationBA(std::vector<cv::Point3f> p3d, std::vector<cv::Point2f> p2d, cv::Mat &R, cv::Mat &t, cv::Mat &cameraMatrix)
{
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    double cost = 0, lastCost = 0;
    double fx = cameraMatrix.at<float>(0, 0);
    double fy = cameraMatrix.at<float>(1, 1);
    double cx = cameraMatrix.at<float>(0, 2);
    double cy = cameraMatrix.at<float>(1, 2);
    Sophus::SE3d pose;

    for (int iter = 0; iter < iterations; iter++) 
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < p3d.size(); i++) 
        {
            Eigen::Vector3d points_3d(p3d[i].x, p3d[i].y, p3d[i].z);
            Eigen::Vector3d pc = pose * points_3d;
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d points_2d(p2d[i].x, p2d[i].y);
            Eigen::Vector2d e = points_2d - proj;

            cost += e.squaredNorm();
            Eigen::Matrix<double, 2, 6> J;
            J << -fx * inv_z,
                0,
                fx * pc[0] * inv_z2,
                fx * pc[0] * pc[1] * inv_z2,
                -fx - fx * pc[0] * pc[0] * inv_z2,
                fx * pc[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pc[1] * inv_z2,
                fy + fy * pc[1] * pc[1] * inv_z2,
                -fy * pc[0] * pc[1] * inv_z2,
                -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
        break;
        }

        if (iter > 0 && cost >= lastCost) {
        // cost increase, update is not good
        break;
        }

        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        if (dx.norm() < 1e-6) {
        // converge
        break;
        }
    }
}




