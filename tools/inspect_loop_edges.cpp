// Offline evidence audit only: no GT or file access is added to the estimator.
#include <loopGeometry.h>
#include <opencv2/core.hpp>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <map>
#include <algorithm>
#include <iomanip>
using namespace superVO;
int main(int argc, char** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error(
                "usage: inspect_loop_edges run-directory image-width image-height");
        cv::setNumThreads(1);
        std::filesystem::path root(argv[1]);
        cv::FileStorage config((root / "run.yaml").string(), cv::FileStorage::READ);
        StereoCalibration c;
        config["K_rectified"] >> c.intrinsics;
        c.imageSize = {std::stoi(argv[2]), std::stoi(argv[3])};
        cv::Mat E;
        config["T_right_left_rectified"] >> E;
        c.baseline = -E.at<double>(0, 3);
        std::map<TimestampNs, cv::Vec3d> gravity;
        std::ifstream inertial(root / "inertial.csv");
        std::string line;
        std::getline(inertial, line);
        std::map<std::string, size_t> column;
        {
            std::istringstream header(line);
            std::string name;
            size_t index = 0;
            while (std::getline(header, name, ','))
                column[name] = index++;
        }
        while (std::getline(inertial, line)) {
            std::istringstream stream(line);
            std::vector<std::string> fields;
            std::string v;
            while (std::getline(stream, v, ','))
                fields.push_back(v);
            gravity[std::stoll(fields.at(column.at("timestamp_ns")))] = {
                std::stod(fields.at(column.at("gx"))), std::stod(fields.at(column.at("gy"))),
                std::stod(fields.at(column.at("gz")))};
        }
        std::vector<LoopKeyframe> keys;
        for (const auto& file : std::filesystem::directory_iterator(root / "loop_keyframes")) {
            cv::FileStorage f(file.path().string(), cv::FileStorage::READ);
            LoopKeyframe key;
            std::string timestamp;
            f["frame"] >> key.frame;
            f["timestamp_ns"] >> timestamp;
            key.timestamp = std::stoll(timestamp);
            key.gravity = gravity.at(key.timestamp);
            cv::Mat pose;
            f["odom_pose"] >> pose;
            key.odomPose = cv::Matx44d(pose);
            auto features = std::make_shared<KeyframeFeatures>();
            f["pixels"] >> features->pixels;
            f["right_pixels"] >> features->rightPixels;
            f["points"] >> features->points;
            f["descriptors"] >> features->descriptors;
            f["depth_valid"] >> features->depthValid;
            key.features = features;
            keys.push_back(key);
        }
        std::sort(keys.begin(), keys.end(),
                  [](auto& a, auto& b) { return a.timestamp < b.timestamp; });
        std::ifstream candidates(root / "loop_candidates.csv");
        std::getline(candidates, line);
        std::cout << "reference_ns,current_ns,reverified,inliers";
        for (int i = 0; i < 16; ++i)
            std::cout << ",T_current_reference_" << i;
        std::cout << '\n';
        while (std::getline(candidates, line)) {
            std::istringstream stream(line);
            std::vector<std::string> fields;
            std::string v;
            while (std::getline(stream, v, ','))
                fields.push_back(v);
            if (fields.size() < 11 || fields[9] != "4")
                continue;
            int i = std::stoi(fields[0]), j = std::stoi(fields[1]);
            cv::setRNGSeed(0);
            auto check = verifyLoop(keys.at(j), keys.at(i), c, {});
            std::cout << keys[j].timestamp << ',' << keys[i].timestamp << ',' << check.valid << ','
                      << check.inliers;
            for (double x : check.currentFromReference.val)
                std::cout << ',' << std::setprecision(15) << x;
            std::cout << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
