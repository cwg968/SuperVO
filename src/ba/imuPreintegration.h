#pragma once
#include <imuTypes.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace superVO::ba {
using V3 = Eigen::Vector3d;
using M3 = Eigen::Matrix3d;
using M15 = Eigen::Matrix<double, 15, 15>;
// Midpoint preintegration in the first IMU frame. Error/covariance order:
// position, velocity, right rotation error, gyro bias, accelerometer bias.
struct ImuPreintegration {
    double dt = 0;
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    V3 position = V3::Zero(), velocity = V3::Zero(), gyroBias, accelBias;
    M3 rotationGyro, positionGyro, positionAccel, velocityGyro, velocityAccel;
    M15 covariance = M15::Zero(), sqrtInformation = M15::Identity();
    V3 gyroNoiseDensity, accelNoiseDensity;
    bool modelUncertainty = false;
    std::vector<ImuSample> samples;
    ImuPreintegration(const std::vector<ImuSample>& samples, const ImuCalibration& calibration,
                      const V3& gyroBias, const V3& accelBias, bool modelUncertainty = false);
};
}
