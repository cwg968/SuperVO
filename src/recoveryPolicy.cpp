#include <recoveryPolicy.h>
#include <cmath>
#include <stdexcept>
namespace superVO {
void RecoveryPolicy::acceptedConstraint(TimestampNs timestamp, bool visual) {
    if (timestamp < lastConstraint_ || timestamp < 0 || phase_ == RecoveryPhase::Lost)
        throw std::logic_error("Invalid recovery constraint timestamp/state");
    lastConstraint_ = timestamp;
    if (visual && phase_ != RecoveryPhase::Tracking) {
        phase_ = ++confirmations_ >= 3 ? RecoveryPhase::Tracking : RecoveryPhase::Confirming;
    }
}
double RecoveryPolicy::sinceConstraint(TimestampNs timestamp) const {
    return lastConstraint_ < 0 ? INFINITY : (timestamp - lastConstraint_) * 1e-9;
}
bool RecoveryPolicy::mayPropagate(TimestampNs timestamp, double positionStd, double rotationStd) {
    const bool valid = lastConstraint_ >= 0 && timestamp >= lastConstraint_ &&
                       sinceConstraint(timestamp) <= maxPropagationSeconds &&
                       std::isfinite(positionStd) && positionStd >= 0 &&
                       positionStd <= maxRelativePositionStd && std::isfinite(rotationStd) &&
                       rotationStd >= 0 && rotationStd <= maxRelativeRotationStd;
    if (phase_ == RecoveryPhase::Lost)
        return false;
    phase_ = valid ? RecoveryPhase::Propagating : RecoveryPhase::Lost;
    confirmations_ = 0;
    return valid;
}
}
