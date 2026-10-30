#pragma once
#include <array>
#include <vector>

namespace superVO {
inline constexpr std::array<const char*, 9> windowAuditGroups{
    "visual_left",        "visual_right",        "imu_white_position",   "imu_white_velocity",
    "imu_white_rotation", "imu_white_gyro_walk", "imu_white_accel_walk", "prior",
    "zero_velocity"};
struct ResidualAudit {
    int blocks = 0, dimensions = 0, invalidBlocks = 0;
    double squaredNorm = 0, robustSquaredNorm = 0, maxAbs = 0;
};
struct ImuWeightAudit {
    int first = -1, second = -1;
    double seconds = 0, whiteningError = 0;
    bool modelUncertainty = false;
    std::array<double, 3> gyroDensity{}, accelDensity{};
    // Covariance order: p,v,right rotation,bg,ba. The full triangular
    // whitener mixes these groups; its diagonal is not an independent weight.
    std::array<double, 15> covarianceDiagonal{}, whitenerDiagonal{};
};
struct WindowAudit {
    int frame = -1;
    std::array<ResidualAudit, 9> residuals{};
    std::vector<ImuWeightAudit> imu;
};
}
