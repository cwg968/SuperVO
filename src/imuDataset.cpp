#include <imuDataset.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace superVO {
namespace {
void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
ImuSample interpolate(const ImuSample& a, const ImuSample& b, TimestampNs t) {
    const double f = double(t - a.timestampNs) / double(b.timestampNs - a.timestampNs);
    return {t, a.angularVelocity * (1 - f) + b.angularVelocity * f,
            a.acceleration * (1 - f) + b.acceleration * f};
}
}
ImuDataset::ImuDataset(const std::filesystem::path& root, const StereoCalibration& cameras) {
    require(cameras.T_body_raw.has_value(), "IMU mode requires camera/body calibration");
    const bool euroc = std::filesystem::is_regular_file(root / "imu0/sensor.yaml");
    const auto dso = root.parent_path() / "dso";
    require(euroc || std::filesystem::is_regular_file(dso / "camchain.yaml"),
            "Missing IMU calibration");
    std::ifstream yaml(euroc ? root / "imu0/sensor.yaml" : dso / "imu_config.yaml");
    require(bool(yaml), "Cannot read IMU calibration");
    std::ostringstream text;
    text << yaml.rdbuf();
    auto source = text.str();
    if (source.rfind("%YAML", 0) != 0)
        source = "%YAML:1.0\n---\n" + source;
    cv::FileStorage fs(source, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    cv::Matx44d bodyFromImu = cv::Matx44d::eye();
    if (euroc) {
        require(std::string(fs["sensor_type"]) == "imu", "Invalid IMU sensor type");
        std::vector<double> matrix;
        fs["T_BS"]["data"] >> matrix;
        require(matrix.size() == 16, "Invalid IMU transform");
        bodyFromImu = cv::Matx44d(matrix.data());
    }
    // TUM VI calibrated export uses IMU as body; do not apply raw-axis or
    // marker corrections a second time. Its camera extrinsics came from Kalibr.
    StereoCalibration check = cameras;
    check.T_body_raw = bodyFromImu;
    check.validate();
    calibration_.T_imu_rect = bodyFromImu.inv(cv::DECOMP_SVD) * *cameras.bodyFromRectified();
    calibration_.gyroNoise = double(fs["gyroscope_noise_density"]);
    calibration_.accelNoise = double(fs["accelerometer_noise_density"]);
    calibration_.gyroWalk = double(fs["gyroscope_random_walk"]);
    calibration_.accelWalk = double(fs["accelerometer_random_walk"]);
    calibration_.rateHz = double(fs[euroc ? "rate_hz" : "update_rate"]);
    calibration_.validate();
    std::ifstream csv(std::filesystem::is_regular_file(root / "imu0/data.csv")
                          ? root / "imu0/data.csv"
                          : dso / "imu.txt");
    require(bool(csv), "Cannot read IMU samples");
    std::string line;
    while (std::getline(csv, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        ImuSample sample;
        std::string extra;
        require(bool(row >> sample.timestampNs >> sample.angularVelocity[0] >>
                     sample.angularVelocity[1] >> sample.angularVelocity[2] >>
                     sample.acceleration[0] >> sample.acceleration[1] >> sample.acceleration[2]) &&
                    !(row >> extra),
                "Malformed IMU row");
        require(sample.timestampNs >= 0 &&
                    (samples_.empty() || sample.timestampNs > samples_.back().timestampNs),
                "IMU timestamps must increase");
        require(cv::checkRange(cv::Mat(sample.angularVelocity)) &&
                    cv::checkRange(cv::Mat(sample.acceleration)),
                "Nonfinite IMU sample");
        samples_.push_back(sample);
    }
    require(samples_.size() >= 2, "Insufficient IMU samples");
}
std::vector<ImuSample> ImuDataset::interval(TimestampNs begin, TimestampNs end) const {
    require(begin <= end && begin >= samples_.front().timestampNs &&
                end <= samples_.back().timestampNs,
            "IMU does not cover image interval");
    auto at = [&](TimestampNs t) {
        auto b =
            std::lower_bound(samples_.begin(), samples_.end(), t,
                             [](const auto& s, TimestampNs time) { return s.timestampNs < time; });
        if (b->timestampNs == t)
            return *b;
        auto a = std::prev(b);
        require((b->timestampNs - a->timestampNs) * 1e-9 <= 4 / calibration_.rateHz,
                "Gap in IMU measurements");
        return interpolate(*a, *b, t);
    };
    std::vector<ImuSample> result{at(begin)};
    auto it = std::upper_bound(samples_.begin(), samples_.end(), begin,
                               [](TimestampNs t, const auto& s) { return t < s.timestampNs; });
    for (; it != samples_.end() && it->timestampNs < end; ++it)
        result.push_back(*it);
    if (end > begin)
        result.push_back(at(end));
    for (size_t i = 1; i < result.size(); ++i)
        require((result[i].timestampNs - result[i - 1].timestampNs) * 1e-9 <=
                    4 / calibration_.rateHz,
                "Gap in IMU measurements");
    return result;
}
}
