#include <tracker.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <set>
namespace {
struct OptionSpec {
    const char* name;
    const char* value;
    const char* description;
    const char* group;
};
constexpr OptionSpec options[] = {
    {"--dataset", "PATH", "Required dataset sequence or mav0 directory", "Input/output"},
    {"--format", "NAME", "auto|euroc|tumvi|kitti (default: auto)", "Input/output"},
    {"--output", "DIR", "Output directory (default: results/run)", "Input/output"},
    {"--image-scale", "FACTOR", "Image/intrinsics scale, 0 < factor <= 1 (default: 1)",
     "Input/output"},
    {"--kitti-cameras", "MODE", "auto|gray|color; KITTI only (default: auto)", "Input/output"},
    {"--inspect", "", "Validate dataset and read its first stereo pair; no inference/output",
     "Input/output"},
    {"--backend", "NAME", "cpu|tensorrt (default: cpu); TensorRT requires --engine", "Inference"},
    {"--weights", "FILE", "CPU .wts model (default: weights/superpoint_v1.wts); CPU only",
     "Inference"},
    {"--engine", "FILE", "TensorRT engine matching padded input dimensions; TensorRT only",
     "Inference"},
    {"--max-points", "N", "Per-image detection/active-point budget (default: 1000)", "Tracking"},
    {"--min-inliers", "N", "Minimum geometric support, N >= 6 (default: 20)", "Tracking"},
    {"--keyframe-interval", "N", "Regular replenishment interval, N >= 1 (default: 10)",
     "Tracking"},
    {"--threshold", "SCORE", "SuperPoint detector threshold, 0 < score < 1 (default: 0.015)",
     "Tracking"},
    {"--imu", "", "Enable IMU; already enabled by superVIO (KITTI has no IMU adapter)", "Inertial"},
    {"--estimator", "NAME", "eskf|window; requires IMU (default: eskf)", "Inertial"},
    {"--observation-noise", "MODEL",
     "uniform|patch; window VIO only (default: patch; recommended: uniform)", "Window VIO"},
    {"--window-states", "N", "State budget, 8..20 (default: 10)", "Window VIO"},
    {"--window-landmarks", "N", "Landmark budget, min-inliers..2000 (default: 300)", "Window VIO"},
    {"--window-iterations", "N", "Main solve limit, 1..30 (default: 6); cleanup has its own limit",
     "Window VIO"},
    {"--window-feedback", "MODE", "off|prune|resolve (default: off; recommended: resolve)",
     "Window VIO"},
    {"--window-audit-every", "N",
     "Read-only residual snapshots every N frames; 0 disables (default: 0)", "Window VIO"},
    {"--visual-ba", "MODE", "none|local (default: none); local BA cannot be combined with IMU",
     "Pure stereo BA"},
    {"--ba-keyframes", "N", "Keyframe budget, 2..12 (default: 6); requires --visual-ba local",
     "Pure stereo BA"},
    {"--ba-landmarks", "N", "Landmark budget, min-inliers..2000 (default: 300)", "Pure stereo BA"},
    {"--ba-iterations", "N", "Solve limit, 1..30 (default: 6)", "Pure stereo BA"},
    {"--loop-closure", "", "Enable gravity-aligned loop closure; requires IMU and loop build",
     "Loop closure"},
    {"--loop-sync", "", "Run loop processing synchronously; requires --loop-closure",
     "Loop closure"},
    {"--viewer", "", "Open Pangolin left-feature and sparse 3D map window", "Display/playback"},
    {"--viewer-no-wait", "", "Close window at EOF; requires --viewer (default: wait for close)",
     "Display/playback"},
    {"--realtime", "", "Pace input by dataset timestamps; does not drop slow input frames",
     "Display/playback"},
    {"--max-frames", "N", "Input-pair limit before IMU overlap clipping; 0 = all (default: 0)",
     "Runtime"},
    {"--threads", "N", "OpenCV CPU thread count, N >= 1 (default: 4)", "Runtime"},
    {"--overwrite", "", "Allow replacing known output files (default: protect existing output)",
     "Runtime"},
    {"--help", "", "Show help (-h is an alias); no dataset required", "Runtime"},
};
void help(bool defaultImu) {
    std::cout << "Usage: " << (defaultImu ? "superVIO" : "superVO") << " --dataset PATH [options]\n"
              << "IMU default: " << (defaultImu ? "enabled" : "disabled")
              << ". Optional features stay disabled unless requested.\n";
    std::string group;
    for (const auto& option : options) {
        if (group != option.group) {
            group = option.group;
            std::cout << '\n' << group << ":\n";
        }
        const auto label =
            std::string(option.name) + (option.value[0] ? std::string(" ") + option.value : "");
        std::cout << "  " << std::left << std::setw(31) << label << option.description << '\n';
    }
    std::cout << "\nWindow VIO requires IMU and SUPERVO_WITH_BA=ON.\n"
              << "See README.markdown for defaults, conditional options and minimal commands.\n";
}
void validate(const superVO::RunOptions& o, const std::set<std::string>& supplied) {
    auto require = [](bool valid, const std::string& message) {
        if (!valid)
            throw std::invalid_argument(message);
    };
    require(!o.dataset.empty(), "--dataset is required; use --help for usage");
    require(o.format == "auto" || o.format == "euroc" || o.format == "tumvi" || o.format == "kitti",
            "--format must be auto, euroc, tumvi or kitti");
    require(o.inference.backend == "cpu" || o.inference.backend == "tensorrt",
            "--backend must be cpu or tensorrt");
    require(o.maxFrames >= 0, "--max-frames must be >= 0");
    require(o.tracking.minInliers >= 6, "--min-inliers must be >= 6");
    require(o.tracking.maxPoints >= o.tracking.minInliers, "--max-points must be >= --min-inliers");
    require(o.threads >= 1, "--threads must be >= 1");
    require(o.tracking.keyframeInterval >= 1, "--keyframe-interval must be >= 1");
    require(std::isfinite(o.tracking.threshold) && o.tracking.threshold > 0 &&
                o.tracking.threshold < 1,
            "--threshold must be in (0, 1)");
    // Inspect validates only data: it deliberately does not initialize model,
    // estimator, loop or viewer resources, even when their flags are present.
    if (o.inspect)
        return;
    require(!o.output.empty(), "--output must not be empty");
    if (o.inference.backend == "tensorrt") {
        require(!o.inference.engine.empty(), "--backend tensorrt requires --engine FILE");
        require(!supplied.count("--weights"),
                "--weights is CPU-only; TensorRT weights are already embedded in --engine");
    } else {
        require(!supplied.count("--engine"), "--engine requires --backend tensorrt");
        require(!o.inference.weights.empty(), "--weights must not be empty");
    }
    require(!supplied.count("--estimator") || o.imu,
            "--estimator requires IMU; use superVIO or --imu");
    const bool window = o.imu && o.vio.backend == superVO::VioBackend::Window;
    for (const auto& flag : supplied) {
        if (flag.rfind("--window-", 0) == 0 || flag == "--observation-noise")
            require(window, flag + " requires IMU and --estimator window");
        if (flag.rfind("--ba-", 0) == 0)
            require(o.stereoBa.enabled, flag + " requires --visual-ba local");
    }
    if (window) {
        require(o.vio.window.states >= 8 && o.vio.window.states <= 20,
                "--window-states must be in [8, 20]");
        require(o.vio.window.landmarks >= o.tracking.minInliers && o.vio.window.landmarks <= 2000,
                "--window-landmarks must be in [--min-inliers, 2000]");
        require(o.vio.window.iterations >= 1 && o.vio.window.iterations <= 30,
                "--window-iterations must be in [1, 30]");
        require(o.vio.window.auditEvery >= 0, "--window-audit-every must be >= 0");
    }
    require(!(o.stereoBa.enabled && o.imu),
            "--visual-ba local cannot be combined with IMU; use superVO");
    if (o.stereoBa.enabled) {
        require(o.stereoBa.keyframes >= 2 && o.stereoBa.keyframes <= 12,
                "--ba-keyframes must be in [2, 12]");
        require(o.stereoBa.landmarks >= o.tracking.minInliers && o.stereoBa.landmarks <= 2000,
                "--ba-landmarks must be in [--min-inliers, 2000]");
        require(o.stereoBa.iterations >= 1 && o.stereoBa.iterations <= 30,
                "--ba-iterations must be in [1, 30]");
    }
    require(!o.loop.enabled || o.imu, "--loop-closure requires IMU; use superVIO or --imu");
    require(!o.loop.synchronous || o.loop.enabled, "--loop-sync requires --loop-closure");
    require(o.viewer.enabled || o.viewer.waitAtEnd, "--viewer-no-wait requires --viewer");
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
        const bool defaultImu = o.imu;
        std::set<std::string> supplied;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h")
                arg = "--help";
            const auto spec = std::find_if(std::begin(options), std::end(options),
                                           [&](const auto& option) { return arg == option.name; });
            if (spec == std::end(options))
                throw std::invalid_argument("Unknown option: " + arg);
            supplied.insert(arg);
            if (arg == "--imu") {
                o.imu = true;
                continue;
            }
            if (arg == "--viewer") {
                o.viewer.enabled = true;
                continue;
            }
            if (arg == "--viewer-no-wait") {
                o.viewer.waitAtEnd = false;
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
            if (arg == "--help") {
                help(defaultImu);
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
            if (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0 ||
                std::string(argv[i + 1]) == "-h")
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
        validate(o, supplied);
        return superVO::runOdometry(o);
    } catch (const std::exception& e) {
        std::cerr << "superVO: " << e.what() << '\n';
        return 1;
    }
}
