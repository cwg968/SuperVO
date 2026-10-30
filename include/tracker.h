#pragma once
#include <trackingTypes.h>
#include <vioOptions.h>
#include <loopGeometry.h>
#include <stereoBaEstimator.h>
#include <viewer.h>
#include <filesystem>
#include <string>
namespace superVO {
struct InferenceOptions {
    std::string backend = "cpu", weights = "weights/superpoint_v1.wts", engine;
};
struct RunOptions {
    std::filesystem::path dataset, output = "results/run";
    std::string format = "auto";
    double imageScale = 1.;
    std::string kittiCameras = "auto";
    TrackingOptions tracking;
    VioOptions vio;
    StereoBaOptions stereoBa;
    LoopOptions loop;
    InferenceOptions inference;
    ViewerOptions viewer;
    int maxFrames = 0, threads = 4;
    bool imu = false;
    bool inspect = false, realtime = false, overwrite = false;
};
int runOdometry(const RunOptions& options);
}
