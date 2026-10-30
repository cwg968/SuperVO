#pragma once
#include <sensorTypes.h>
#include <vector>
namespace superVO {
struct ImuSample {
    TimestampNs timestampNs = 0;
    cv::Vec3d angularVelocity{}, acceleration{}; // rad/s and specific force m/s^2, IMU axes
};
struct ImuCalibration {
    cv::Matx44d T_imu_rect = cv::Matx44d::eye();
    double gyroNoise = 0, accelNoise = 0, gyroWalk = 0, accelWalk = 0, rateHz = 0;
    void validate() const;
};
void validateImuInterval(const std::vector<ImuSample>& samples, TimestampNs begin, TimestampNs end,
                         double rateHz);
struct InertialState {
    bool initialized = false;
    cv::Matx44d T_world_rect = cv::Matx44d::eye();
    cv::Vec3d velocity{}, gravity{}, gyroBias{}, accelBias{};
    double positionStd = 0, innovation = 0;
};
}
