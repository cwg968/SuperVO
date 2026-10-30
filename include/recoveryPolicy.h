#pragma once
#include <sensorTypes.h>
namespace superVO {
enum class RecoveryPhase { Tracking, Propagating, Confirming, Lost };
// Sensor-independent recovery decisions. New stereo depth is not a motion constraint.
class RecoveryPolicy {
  public:
    static constexpr double maxPropagationSeconds = 2.;
    static constexpr double maxRelativePositionStd = .35;
    static constexpr double maxRelativeRotationStd = .12;
    void acceptedConstraint(TimestampNs timestamp, bool visual);
    bool mayPropagate(TimestampNs timestamp, double positionStd, double rotationStd);
    RecoveryPhase phase() const {
        return phase_;
    }
    double sinceConstraint(TimestampNs timestamp) const;

  private:
    TimestampNs lastConstraint_ = -1;
    int confirmations_ = 0;
    RecoveryPhase phase_ = RecoveryPhase::Tracking;
};
}
