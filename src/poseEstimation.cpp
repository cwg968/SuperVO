#include <poseEstimation.h>
#include <opencv2/calib3d.hpp>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace superVO {
PoseEstimate estimatePose(const std::vector<cv::Point3f>& points,
                          const std::vector<cv::Point2f>& pixels, const cv::Mat& K, int minInliers,
                          double threshold) {
    PoseEstimate result;
    if (minInliers < 6 || points.size() != pixels.size() ||
        points.size() < static_cast<size_t>(minInliers) || K.rows != 3 || K.cols != 3 ||
        threshold <= 0)
        return result;
    for (size_t i = 0; i < points.size(); ++i)
        if (!std::isfinite(points[i].x) || !std::isfinite(points[i].y) ||
            !std::isfinite(points[i].z) || !std::isfinite(pixels[i].x) ||
            !std::isfinite(pixels[i].y))
            return result;
    try {
        cv::Mat r, t, mask;
        if (!cv::solvePnPRansac(points, pixels, K, cv::noArray(), r, t, false, 100, threshold,
                                0.999, mask, cv::SOLVEPNP_EPNP) ||
            mask.rows < minInliers)
            return result;
        std::vector<cv::Point3f> good3;
        std::vector<cv::Point2f> good2;
        for (int j = 0; j < mask.rows; ++j) {
            int i = mask.at<int>(j);
            good3.push_back(points.at(i));
            good2.push_back(pixels.at(i));
        }
        cv::solvePnPRefineLM(good3, good2, K, cv::noArray(), r, t);
        if (!cv::checkRange(r) || !cv::checkRange(t))
            return result;
        cv::Mat R;
        cv::Rodrigues(r, R);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col)
                result.T_camera_reference(row, col) = R.at<double>(row, col);
            result.T_camera_reference(row, 3) = t.at<double>(row);
        }
        std::vector<cv::Point2f> projected;
        cv::projectPoints(points, r, t, K, cv::noArray(), projected);
        double error = 0;
        for (size_t j = 0; j < points.size(); ++j) {
            const auto& p = points[j];
            double depth = R.at<double>(2, 0) * p.x + R.at<double>(2, 1) * p.y +
                           R.at<double>(2, 2) * p.z + t.at<double>(2);
            double e = cv::norm(projected[j] - pixels[j]);
            if (depth > 0 && std::isfinite(e) && e <= threshold) {
                result.inliers.push_back(static_cast<int>(j));
                error += e * e;
            }
        }
        result.valid = result.inliers.size() >= static_cast<size_t>(minInliers);
        if (result.valid)
            result.rmse = std::sqrt(error / result.inliers.size());
    } catch (const cv::Exception&) {
        return PoseEstimate{};
    }
    return result;
}
cv::Matx44d leftPoseFromRight(const cv::Matx44d& rightPose, const cv::Matx44d& rightFromLeft) {
    return rightFromLeft.inv(cv::DECOMP_SVD) * rightPose;
}
StereoPoseQuality assessStereoPose(const cv::Matx44d& leftPose,
                                   const std::vector<cv::Point3f>& points,
                                   const std::array<PointTracks, 2>& tracks, const cv::Mat& K,
                                   const cv::Matx44d& rightFromLeft, cv::Size size,
                                   double threshold) {
    StereoPoseQuality result;
    if (size.width <= 0 || size.height <= 0 || threshold <= 0 || !cv::checkRange(cv::Mat(leftPose)))
        return result;
    std::vector<double> contribution(points.size(), 0);
    std::vector<int> observed(points.size(), 0);
    std::vector<bool> supported(points.size(), false);
    double squaredError = 0;
    size_t measurements = 0;
    int cameraCount = 0;
    const double fx = K.at<double>(0, 0), fy = K.at<double>(1, 1), cx = K.at<double>(0, 2),
                 cy = K.at<double>(1, 2);
    for (int c = 0; c < 2; ++c) {
        if (tracks[c].pixels.size() != tracks[c].sourceIndices.size())
            throw std::invalid_argument("Invalid camera observations");
        const auto transform = c ? rightFromLeft * leftPose : leftPose;
        std::array<bool, 24> cells{};
        std::vector<bool> seen(points.size(), false);
        for (size_t j = 0; j < tracks[c].pixels.size(); ++j) {
            const size_t id = tracks[c].sourceIndices[j];
            if (id >= points.size() || seen[id])
                throw std::invalid_argument("Invalid or duplicated landmark index");
            seen[id] = true;
            ++observed[id];
            const auto& p = points[id];
            const auto& pixel = tracks[c].pixels[j];
            const auto q = transform * cv::Vec4d(p.x, p.y, p.z, 1);
            if (q[2] <= 0 || !std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2]) ||
                !std::isfinite(pixel.x) || !std::isfinite(pixel.y) || pixel.x < 0 || pixel.y < 0 ||
                pixel.x >= size.width || pixel.y >= size.height)
                continue;
            const double error = cv::norm(
                cv::Point2d(fx * q[0] / q[2] + cx, fy * q[1] / q[2] + cy) - cv::Point2d(pixel));
            if (!std::isfinite(error) || error > threshold)
                continue;
            contribution[id] += std::max(0., 1 - error * error / (threshold * threshold));
            supported[id] = true;
            result.inlierObservations[c].push_back(int(j));
            cells[int(pixel.y * 4 / size.height) * 6 + int(pixel.x * 6 / size.width)] = true;
            squaredError += error * error;
            ++measurements;
        }
        if (!tracks[c].pixels.empty()) {
            result.coverage += std::count(cells.begin(), cells.end(), true) / 24.;
            ++cameraCount;
        }
    }
    for (size_t j = 0; j < points.size(); ++j) {
        result.uniqueInliers += supported[j];
        if (observed[j])
            result.score += contribution[j] / observed[j];
    }
    if (cameraCount)
        result.coverage /= cameraCount;
    result.score *= .5 + .5 * result.coverage;
    if (measurements)
        result.rmse = std::sqrt(squaredError / measurements);
    return result;
}
int selectStereoPose(const std::array<PoseEstimate, 2>& candidates,
                     const std::array<StereoPoseQuality, 2>& quality, int minimum) {
    int selected = -1;
    for (int c = 0; c < 2; ++c) {
        if (!candidates[c].valid || candidates[c].inliers.size() < static_cast<size_t>(minimum) ||
            quality[c].uniqueInliers < static_cast<size_t>(minimum) ||
            !std::isfinite(quality[c].score) || quality[c].score <= 0)
            continue;
        if (selected < 0 ||
            quality[c].score > quality[selected].score * (1 + stereoPoseScoreMargin) + 1e-9)
            selected = c;
    }
    return selected;
}
cv::Matx44d cameraPose(const cv::Matx44d& reference, const cv::Matx44d& relative) {
    return reference * relative.inv(cv::DECOMP_SVD);
}

}
void poseEstimation3d2d(std::vector<cv::Point3f> p3, std::vector<cv::Point2f> p2, cv::Mat& R,
                        cv::Mat& t, cv::Mat& K) {
    auto estimate = superVO::estimatePose(p3, p2, K, 6);
    if (!estimate.valid)
        throw std::runtime_error("PnP failed: insufficient valid inliers");
    cv::Mat T(estimate.T_camera_reference);
    R = T(cv::Rect(0, 0, 3, 3)).clone();
    t = T(cv::Rect(3, 0, 1, 3)).clone();
}
void poseEstimation2d2d(std::vector<cv::Point2f> a, std::vector<cv::Point2f> b, cv::Mat& R,
                        cv::Mat& t, cv::Mat& K) {
    if (a.size() < 8 || a.size() != b.size())
        throw std::runtime_error("Insufficient essential-matrix correspondences");
    cv::Mat mask;
    auto E = cv::findEssentialMat(a, b, K, cv::RANSAC, 0.999, 1.0, mask);
    if (E.empty() || cv::recoverPose(E, a, b, K, R, t, mask) < 5)
        throw std::runtime_error("Essential-matrix pose recovery failed");
}
