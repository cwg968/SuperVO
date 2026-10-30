#pragma once
#include <sensorTypes.h>
namespace superVO {
// Constant body-frame rotation/translation rates. Used only to initialize matching.
cv::Matx44d predictCameraPose(const cv::Matx44d& before, const cv::Matx44d& last,
                              double previousSeconds, double nextSeconds);
// Owns only accepted pose history. Its output is never an observation or final pose.
class ConstantVelocityPredictor {
  public:
    cv::Matx44d predict(TimestampNs timestampNs) const;
    void accept(const cv::Matx44d& pose, TimestampNs timestampNs);
    // Re-express accepted history after local BA; a measurement is not added.
    void rebase(const cv::Matx44d& correction);
    const cv::Matx44d& lastPose() const {
        return last_;
    }

  private:
    cv::Matx44d before_ = cv::Matx44d::eye(), last_ = cv::Matx44d::eye();
    double previousSeconds_ = 0;
    TimestampNs timestampNs_ = 0;
    bool initialized_ = false;
};
}
