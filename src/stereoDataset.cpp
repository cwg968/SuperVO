#include <stereoDataset.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cmath>
#include <stdexcept>

namespace superVO {
namespace {
void check(bool value, const std::string& message) {
    if (!value)
        throw std::runtime_error(message);
}
std::string text(const std::filesystem::path& path) {
    std::ifstream in(path);
    check(bool(in), "Cannot read " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}
cv::FileStorage yaml(const std::filesystem::path& path, bool kalibr = false) {
    std::string source = text(path);
    // PyYAML/Kalibr permits indentless block sequences; OpenCV's YAML reader
    // requires sequence entries to be indented beneath their mapping key.
    if (kalibr) {
        std::istringstream lines(source);
        std::ostringstream normalized;
        std::string line;
        while (std::getline(lines, line)) {
            auto first = line.find_first_not_of(' ');
            if (first != std::string::npos && line.compare(first, 2, "- ") == 0)
                line.insert(0, "  ");
            normalized << line << '\n';
        }
        source = normalized.str();
    }
    if (source.compare(0, 5, "%YAML") != 0)
        source = "%YAML:1.0\n---\n" + source;
    return cv::FileStorage(source, cv::FileStorage::READ | cv::FileStorage::MEMORY);
}
cv::Matx44d checkedTransform(const std::vector<double>& data) {
    check(data.size() == 16, "Expected a 4x4 T_BS transform");
    cv::Matx44d T(data.data());
    cv::Matx33d R(T(0, 0), T(0, 1), T(0, 2), T(1, 0), T(1, 1), T(1, 2), T(2, 0), T(2, 1), T(2, 2));
    check(cv::checkRange(cv::Mat(T)) && cv::norm(cv::Mat(R.t() * R - cv::Matx33d::eye())) < 1e-4 &&
              std::abs(cv::determinant(R) - 1) < 1e-4 &&
              cv::norm(cv::Vec4d(T(3, 0), T(3, 1), T(3, 2), T(3, 3)) - cv::Vec4d(0, 0, 0, 1)) <
                  1e-9,
          "Invalid T_BS rigid transform");
    return T;
}
cv::Matx44d transform(const cv::FileNode& node) {
    std::vector<double> data;
    node["data"] >> data;
    return checkedTransform(data);
}
struct Camera {
    cv::Mat K, D;
    cv::Size size;
    cv::Matx44d T;
    bool fisheye = false;
};
Camera cameraNode(const cv::FileNode& node, bool kalibr) {
    const auto distortion = std::string(node["distortion_model"]);
    check(std::string(node["camera_model"]) == "pinhole" &&
              (distortion == "radial-tangential" || distortion == "equidistant"),
          "Unsupported camera model");
    std::vector<double> k, d;
    std::vector<int> resolution;
    node["intrinsics"] >> k;
    node[kalibr ? "distortion_coeffs" : "distortion_coefficients"] >> d;
    node["resolution"] >> resolution;
    check(k.size() == 4 && d.size() == 4 && resolution.size() == 2,
          "Incomplete camera calibration");
    check(k[0] > 0 && k[1] > 0 && resolution[0] > 0 && resolution[1] > 0,
          "Invalid camera calibration");
    Camera c;
    c.K = (cv::Mat_<double>(3, 3) << k[0], 0, k[2], 0, k[1], k[3], 0, 0, 1);
    c.D = cv::Mat(d).clone();
    c.size = {resolution[0], resolution[1]};
    c.fisheye = distortion == "equidistant";
    if (kalibr) {
        const auto matrix = node["T_cam_imu"];
        check(matrix.isSeq() && matrix.size() == 4, "Expected 4x4 T_cam_imu");
        std::vector<double> values;
        for (const auto& row : matrix) {
            std::vector<double> v;
            row >> v;
            check(v.size() == 4, "Expected 4x4 T_cam_imu");
            values.insert(values.end(), v.begin(), v.end());
        }
        // Kalibr maps IMU into camera, whereas geometry stores body-from-camera.
        c.T = checkedTransform(values).inv(cv::DECOMP_SVD);
    } else
        c.T = transform(node["T_BS"]);
    check(cv::checkRange(c.K) && cv::checkRange(c.D), "Non-finite camera calibration");
    return c;
}
Camera camera(const std::filesystem::path& path) {
    auto fs = yaml(path);
    return cameraNode(fs.root(), false);
}
std::map<int64_t, std::string> imageList(const std::filesystem::path& path) {
    std::ifstream in(path);
    check(bool(in), "Cannot read " + path.string());
    std::map<int64_t, std::string> rows;
    std::string line;
    int64_t previous = -1;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line[0] == '#')
            continue;
        auto comma = line.find(',');
        check(comma != std::string::npos, "Invalid image CSV row");
        size_t used = 0;
        auto stamp = std::stoll(line.substr(0, comma), &used);
        check(used == comma && stamp >= 0 && stamp > previous,
              "Image timestamps must be strictly increasing");
        auto name = line.substr(comma + 1);
        check(!name.empty() && std::filesystem::path(name).filename() == name,
              "Invalid image filename");
        rows.emplace(stamp, name);
        previous = stamp;
    }
    check(!rows.empty(), "Empty camera CSV: " + path.string());
    return rows;
}
}
StereoDataset::StereoDataset(const std::filesystem::path& path, const std::string& format,
                             double imageScale, const std::string& kittiCameras)
    : root_(std::filesystem::absolute(path)), format_(format), imageScale_(imageScale),
      kittiCameras_(kittiCameras) {
    check(std::isfinite(imageScale_) && imageScale_ > 0 && imageScale_ <= 1,
          "Image scale must be in (0, 1]");
    check(kittiCameras_ == "auto" || kittiCameras_ == "gray" || kittiCameras_ == "color",
          "Invalid KITTI camera selection");
    if (std::filesystem::exists(root_ / "mav0" / "cam0"))
        root_ /= "mav0";
    if (format_ == "auto") {
        if (std::filesystem::exists(root_.parent_path() / "dso/camchain.yaml") &&
            std::filesystem::exists(root_ / "cam0/data.csv"))
            format_ = "tumvi";
        else
            format_ = std::filesystem::exists(root_ / "cam0" / "sensor.yaml") ? "euroc" : "kitti";
    }
    if (format_ == "euroc" || format_ == "tumvi")
        loadEuroc();
    else if (format_ == "kitti")
        loadKitti();
    else
        throw std::runtime_error("Dataset format must be auto, euroc, tumvi or kitti");
    sourceSize_ = size_;
    size_ = {cvRound(sourceSize_.width * imageScale_), cvRound(sourceSize_.height * imageScale_)};
    check(size_.width >= 8 && size_.height >= 8, "Scaled images must be at least 8x8");
    if (size_ != sourceSize_) {
        const double sx = double(size_.width) / sourceSize_.width,
                     sy = double(size_.height) / sourceSize_.height;
        // cv::resize maps pixel centers: u' = s * (u + .5) - .5.
        const cv::Mat A =
            (cv::Mat_<double>(3, 3) << sx, 0, .5 * sx - .5, 0, sy, .5 * sy - .5, 0, 0, 1);
        K_ = A * K_;
    }
    check(!samples_.empty(), "Dataset is empty");
    check(std::isfinite(baseline_) && baseline_ > 0, "Invalid stereo baseline");
    for (const auto& sample : samples_)
        check(std::filesystem::is_regular_file(sample.left) &&
                  std::filesystem::is_regular_file(sample.right),
              "Missing stereo image at timestamp " + std::to_string(sample.timestampNs));
}
void StereoDataset::loadEuroc() {
    Camera left, right;
    if (format_ == "tumvi") {
        auto fs = yaml(root_.parent_path() / "dso/camchain.yaml", true);
        left = cameraNode(fs["cam0"], true);
        right = cameraNode(fs["cam1"], true);
    } else {
        left = camera(root_ / "cam0/sensor.yaml");
        right = camera(root_ / "cam1/sensor.yaml");
    }
    check(left.fisheye == right.fisheye, "Mixed stereo camera distortion models are unsupported");
    check(left.size == right.size, "Stereo image sizes differ");
    size_ = left.size;
    T_body_camera_ = left.T;
    cv::Matx44d relative = right.T.inv() * left.T;
    cv::Mat relativeMat(relative), R0, R1, P0, P1, Q;
    if (left.fisheye)
        cv::fisheye::stereoRectify(left.K, left.D, right.K, right.D, size_,
                                   relativeMat(cv::Rect(0, 0, 3, 3)),
                                   relativeMat(cv::Rect(3, 0, 1, 3)).clone(), R0, R1, P0, P1, Q,
                                   cv::CALIB_ZERO_DISPARITY, size_, 0., 1.);
    else
        cv::stereoRectify(left.K, left.D, right.K, right.D, size_,
                          relativeMat(cv::Rect(0, 0, 3, 3)), relativeMat(cv::Rect(3, 0, 1, 3)), R0,
                          R1, P0, P1, Q, cv::CALIB_ZERO_DISPARITY, 0, size_);
    K_ = P0(cv::Rect(0, 0, 3, 3)).clone();
    baseline_ = -P1.at<double>(0, 3) / K_.at<double>(0, 0);
    check(std::abs(P1.at<double>(1, 3)) < 1e-6, "Only horizontal rectified stereo is supported");
    R_rect_ = cv::Matx33d(R0);
    if (left.fisheye) {
        cv::fisheye::initUndistortRectifyMap(left.K, left.D, R0, P0, size_, CV_32FC1, maps_[0],
                                             maps_[1]);
        cv::fisheye::initUndistortRectifyMap(right.K, right.D, R1, P1, size_, CV_32FC1, maps_[2],
                                             maps_[3]);
    } else {
        cv::initUndistortRectifyMap(left.K, left.D, R0, P0, size_, CV_32FC1, maps_[0], maps_[1]);
        cv::initUndistortRectifyMap(right.K, right.D, R1, P1, size_, CV_32FC1, maps_[2], maps_[3]);
    }
    auto a = imageList(root_ / "cam0" / "data.csv"), b = imageList(root_ / "cam1" / "data.csv");
    for (const auto& row : a) {
        auto match = b.find(row.first);
        if (match == b.end()) {
            ++unpairedLeft_;
            continue;
        }
        samples_.push_back({row.first, root_ / "cam0" / "data" / row.second,
                            root_ / "cam1" / "data" / match->second});
    }
    unpairedRight_ = b.size() - samples_.size();
}
void StereoDataset::loadKitti() {
    std::ifstream calibration(root_ / "calib.txt");
    check(bool(calibration), "Cannot read calib.txt");
    cv::Mat P[4];
    std::string line;
    while (std::getline(calibration, line)) {
        std::istringstream row(line);
        std::string label;
        row >> label;
        if (label != "P0:" && label != "P1:" && label != "P2:" && label != "P3:")
            continue;
        int i = label[1] - '0';
        P[i] = cv::Mat(3, 4, CV_64F);
        for (int j = 0; j < 12; ++j)
            check(bool(row >> P[i].at<double>(j / 4, j % 4)), "Malformed KITTI calibration");
    }
    leftCamera_ = kittiCameras_ == "color" || (kittiCameras_ == "auto" &&
                                               !std::filesystem::is_directory(root_ / "image_0"))
                      ? 2
                      : 0;
    const int l = leftCamera_, r = l + 1;
    check(!P[l].empty() && !P[r].empty(), "Selected KITTI camera calibration missing");
    K_ = P[l](cv::Rect(0, 0, 3, 3)).clone();
    const cv::Mat rightK = P[r](cv::Rect(0, 0, 3, 3)).clone();
    check(cv::checkRange(P[l]) && cv::checkRange(P[r]) && K_.at<double>(0, 0) > 0 &&
              K_.at<double>(1, 1) > 0 && rightK.at<double>(0, 0) > 0 && rightK.at<double>(1, 1) > 0,
          "Invalid KITTI calibration");
    cv::Mat t0 = K_.inv() * P[l].col(3), t1 = rightK.inv() * P[r].col(3);
    if (l == 2) {
        check(!P[0].empty() && cv::checkRange(P[0]),
              "KITTI P0 needed for color-camera output coordinates");
        const cv::Mat origin = P[0](cv::Rect(0, 0, 3, 3)).inv() * P[0].col(3) - t0;
        for (int axis = 0; axis < 3; ++axis)
            T_cam0_left_(axis, 3) = origin.at<double>(axis);
    }
    baseline_ = t0.at<double>(0) - t1.at<double>(0);
    std::ifstream times(root_ / "times.txt");
    check(bool(times), "Cannot read times.txt");
    int index = 0;
    int64_t previous = -1;
    while (std::getline(times, line)) {
        if (line.empty())
            continue;
        std::istringstream row(line);
        double seconds;
        std::string extra;
        check(bool(row >> seconds) && !(row >> extra) && std::isfinite(seconds) && seconds >= 0 &&
                  seconds < 1e9,
              "Invalid KITTI timestamp");
        int64_t stamp = std::llround(seconds * 1e9);
        check(stamp > previous, "KITTI timestamps must increase");
        previous = stamp;
        std::ostringstream name;
        name << std::setw(6) << std::setfill('0') << index++ << ".png";
        samples_.push_back({stamp, root_ / ("image_" + std::to_string(l)) / name.str(),
                            root_ / ("image_" + std::to_string(r)) / name.str()});
    }
    check(!samples_.empty(), "Empty KITTI sequence");
    auto first = cv::imread(samples_[0].left.string(), cv::IMREAD_GRAYSCALE);
    check(!first.empty(), "Cannot read first KITTI image");
    size_ = first.size();
    if (l == 0) {
        check(cv::norm(K_ - rightK) < 1e-3, "KITTI grayscale cameras must have equal intrinsics");
        check(std::abs(t1.at<double>(1) - t0.at<double>(1)) < 1e-5 &&
                  std::abs(t1.at<double>(2) - t0.at<double>(2)) < 1e-5,
              "Unsupported non-horizontal KITTI grayscale stereo");
    } else {
        // Color P2/P3 contain small y/z offsets. Re-rectify their already
        // distortion-corrected images rather than silently discarding offsets.
        cv::Mat R0, R1, P0, P1, Q;
        const cv::Mat D = cv::Mat::zeros(1, 5, CV_64F);
        cv::stereoRectify(K_, D, rightK, D, size_, cv::Mat::eye(3, 3, CV_64F), t1 - t0, R0, R1, P0,
                          P1, Q, cv::CALIB_ZERO_DISPARITY, 0, size_);
        cv::initUndistortRectifyMap(K_, D, R0, P0, size_, CV_32FC1, maps_[0], maps_[1]);
        cv::initUndistortRectifyMap(rightK, D, R1, P1, size_, CV_32FC1, maps_[2], maps_[3]);
        K_ = P0(cv::Rect(0, 0, 3, 3)).clone();
        baseline_ = -P1.at<double>(0, 3) / K_.at<double>(0, 0);
        R_rect_ = cv::Matx33d(R0);
    }
}
StereoCalibration StereoDataset::calibration() const {
    StereoCalibration result;
    result.intrinsics = K_.clone();
    result.imageSize = size_;
    result.baseline = baseline_;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            result.T_rect_raw(r, c) = R_rect_(r, c);
    if (format_ == "euroc" || format_ == "tumvi")
        result.T_body_raw = T_body_camera_;
    return result;
}
StereoImages StereoDataset::read(size_t index) const {
    const auto& s = samples_.at(index);
    StereoImages images{cv::imread(s.left.string(), cv::IMREAD_GRAYSCALE),
                        cv::imread(s.right.string(), cv::IMREAD_GRAYSCALE)};
    check(!images.left.empty() && !images.right.empty(), "Cannot decode stereo images");
    check(images.left.size() == sourceSize_ && images.right.size() == sourceSize_,
          "Image dimensions disagree with calibration");
    if (!maps_[0].empty()) {
        cv::remap(images.left, images.left, maps_[0], maps_[1], cv::INTER_LINEAR);
        cv::remap(images.right, images.right, maps_[2], maps_[3], cv::INTER_LINEAR);
    }
    if (size_ != sourceSize_) {
        cv::resize(images.left, images.left, size_, 0, 0, cv::INTER_AREA);
        cv::resize(images.right, images.right, size_, 0, 0, cv::INTER_AREA);
    }
    return images;
}
}
