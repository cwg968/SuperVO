#include <tracker.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
namespace {
void help() {
    std::cout
        << "Usage: superVO --dataset PATH [--format auto|euroc|tumvi|kitti] [--output DIR]\n"
        << "  --weights FILE       SuperPoint .wts weights (default weights/superpoint_v1.wts)\n"
        << "  --backend cpu|tensorrt --engine FILE\n"
        << "  --image-scale FACTOR  Downsample images and intrinsics (0 < factor <= 1)\n"
        << "  --kitti-cameras auto|gray|color  Select image_0/1 or image_2/3\n"
        << "  Pipeline: SuperPoint stereo + epipolar refinement; left/right temporal LK + scored PnP (left-camera trajectory)\n"
        << "  --max-frames N       Limit sequence length (0 = all)\n"
        << "  --max-points N --min-inliers N --keyframe-interval N --threshold SCORE\n"
        << "  --imu              Enable stereo visual-inertial fusion (EuRoC/TUM VI)\n"
        << "  --estimator eskf|window --observation-noise uniform|patch\n"
        << "  --window-states N --window-landmarks N --window-iterations N\n"
        << "  --window-feedback off|prune|resolve  Publish posterior observation status and prune persistent outliers\n"
        << "  --window-audit-every N  Read-only factor snapshots every N frames (0 disables)\n"
        << "  --visual-ba none|local  Pure stereo local BA (without IMU)\n"
        << "  --ba-keyframes N --ba-landmarks N --ba-iterations N\n"
        << "  --loop-closure --loop-sync  Enable global loop correction (sync for deterministic evaluation)\n"
        << "  --threads N --inspect --realtime --overwrite --help\n";
}
int integer(const std::string& value) {
    size_t used;
    int n = std::stoi(value, &used);
    if (used != value.size())
        throw std::runtime_error("Invalid integer: " + value);
    return n;
}
}
int main(int argc, char** argv) {
    try {
        superVO::RunOptions o;
#ifdef SUPERVO_DEFAULT_IMU
        o.imu = true;
#endif
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--imu") {
                o.imu = true;
                continue;
            }
            if (arg == "--loop-closure") {
                o.loop.enabled = true;
                continue;
            }
            if (arg == "--loop-sync") {
                o.loop.synchronous = true;
                continue;
            }
            if (arg == "--help" || arg == "-h") {
                help();
                return 0;
            }
            if (arg == "--inspect") {
                o.inspect = true;
                continue;
            }
            if (arg == "--realtime") {
                o.realtime = true;
                continue;
            }
            if (arg == "--overwrite") {
                o.overwrite = true;
                continue;
            }
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + arg);
            std::string value = argv[++i];
            if (arg == "--dataset")
                o.dataset = value;
            else if (arg == "--output")
                o.output = value;
            else if (arg == "--format")
                o.format = value;
            else if (arg == "--image-scale") {
                size_t used = 0;
                o.imageScale = std::stod(value, &used);
                if (used != value.size() || !std::isfinite(o.imageScale) || o.imageScale <= 0 ||
                    o.imageScale > 1)
                    throw std::runtime_error("Image scale must be in (0, 1]");
            } else if (arg == "--kitti-cameras") {
                if (value != "auto" && value != "gray" && value != "color")
                    throw std::runtime_error("KITTI cameras must be auto, gray or color");
                o.kittiCameras = value;
            } else if (arg == "--backend")
                o.inference.backend = value;
            else if (arg == "--estimator") {
                if (value != "eskf" && value != "window")
                    throw std::runtime_error("Estimator must be eskf or window");
                o.vio.backend =
                    value == "window" ? superVO::VioBackend::Window : superVO::VioBackend::Eskf;
            } else if (arg == "--visual-ba") {
                if (value != "none" && value != "local")
                    throw std::runtime_error("Visual BA must be none or local");
                o.stereoBa.enabled = value == "local";
            } else if (arg == "--ba-keyframes")
                o.stereoBa.keyframes = integer(value);
            else if (arg == "--ba-landmarks")
                o.stereoBa.landmarks = integer(value);
            else if (arg == "--ba-iterations")
                o.stereoBa.iterations = integer(value);
            else if (arg == "--observation-noise") {
                if (value != "uniform" && value != "patch")
                    throw std::runtime_error("Observation noise must be uniform or patch");
                o.vio.noise = value == "patch" ? superVO::ObservationNoise::Patch
                                               : superVO::ObservationNoise::Uniform;
            } else if (arg == "--window-states")
                o.vio.window.states = integer(value);
            else if (arg == "--window-landmarks")
                o.vio.window.landmarks = integer(value);
            else if (arg == "--window-iterations")
                o.vio.window.iterations = integer(value);
            else if (arg == "--window-audit-every")
                o.vio.window.auditEvery = integer(value);
            else if (arg == "--window-feedback") {
                if (value != "off" && value != "prune" && value != "resolve")
                    throw std::runtime_error("Window feedback must be off, prune or resolve");
                o.vio.window.observationFeedback = value != "off";
                o.vio.window.resolveAfterPruning = value == "resolve";
            } else if (arg == "--weights")
                o.inference.weights = value;
            else if (arg == "--engine")
                o.inference.engine = value;
            else if (arg == "--max-frames")
                o.maxFrames = integer(value);
            else if (arg == "--max-points")
                o.tracking.maxPoints = integer(value);
            else if (arg == "--min-inliers")
                o.tracking.minInliers = integer(value);
            else if (arg == "--keyframe-interval")
                o.tracking.keyframeInterval = integer(value);
            else if (arg == "--threads")
                o.threads = integer(value);
            else if (arg == "--threshold") {
                size_t used;
                o.tracking.threshold = std::stof(value, &used);
                if (used != value.size())
                    throw std::runtime_error("Invalid threshold");
            } else
                throw std::runtime_error("Unknown option: " + arg);
        }
        if (o.dataset.empty())
            throw std::runtime_error("--dataset is required; use --help for usage");
        if (o.maxFrames < 0 || o.tracking.minInliers < 6 ||
            o.tracking.maxPoints < o.tracking.minInliers || o.threads < 1 ||
            o.tracking.keyframeInterval < 1 || !std::isfinite(o.tracking.threshold) ||
            o.tracking.threshold <= 0 || o.tracking.threshold >= 1)
            throw std::runtime_error("Invalid tracking configuration");
        if (o.inference.backend != "cpu" && o.inference.backend != "tensorrt")
            throw std::runtime_error("Backend must be cpu or tensorrt");
        if (o.stereoBa.enabled && o.imu)
            throw std::runtime_error("Pure visual BA cannot be combined with an IMU estimator");
        return superVO::runOdometry(o);
    } catch (const std::exception& e) {
        std::cerr << "superVO: " << e.what() << '\n';
        return 1;
    }
}
