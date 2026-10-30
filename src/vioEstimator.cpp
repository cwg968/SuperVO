#include <vioEstimator.h>
#include <stdexcept>
namespace superVO {
struct VioEstimator::Impl {
    std::optional<InertialEstimator> filter;
#ifdef SUPERVO_WITH_BA
    std::optional<WindowEstimator> window;
#endif
    Impl(StereoCalibration camera, ImuCalibration imu, VioOptions options) {
        if (options.backend == VioBackend::Eskf)
            filter.emplace(imu);
        else {
#ifdef SUPERVO_WITH_BA
            window.emplace(std::move(camera), imu, options.window);
#else
            (void)camera;
            throw std::invalid_argument("Window VIO requires a build with SUPERVO_WITH_BA=ON");
#endif
        }
    }
};
VioEstimator::VioEstimator(StereoCalibration c, ImuCalibration i, VioOptions o)
    : impl_(std::make_unique<Impl>(std::move(c), i, o)) {}
VioEstimator::~VioEstimator() = default;
VioEstimator::VioEstimator(const VioEstimator& o) : impl_(std::make_unique<Impl>(*o.impl_)) {}
VioEstimator& VioEstimator::operator=(const VioEstimator& o) {
    if (this != &o)
        impl_ = std::make_unique<Impl>(*o.impl_);
    return *this;
}
bool VioEstimator::usesPixels() const {
    return !impl_->filter;
}
bool VioEstimator::initialized() const {
    if (impl_->filter)
        return impl_->filter->initialized();
#ifdef SUPERVO_WITH_BA
    return impl_->window->initialized();
#else
    return false;
#endif
}
void VioEstimator::initialize(const InertialState& state, TimestampNs timestamp, int frame) {
    if (impl_->filter)
        impl_->filter->initialize(state, timestamp);
#ifdef SUPERVO_WITH_BA
    else
        impl_->window->initialize(state, timestamp, frame);
#else
    (void)frame;
#endif
}
void VioEstimator::propagate(const std::vector<ImuSample>& interval) {
    if (impl_->filter)
        impl_->filter->propagate(interval);
#ifdef SUPERVO_WITH_BA
    else
        impl_->window->propagate(interval);
#endif
}
bool VioEstimator::correct(const cv::Matx44d& transform, int reference, double rmse, size_t inliers,
                           const VisualMeasurements& batch) {
    if (impl_->filter)
        return impl_->filter->correct(transform, reference, rmse, inliers);
#ifdef SUPERVO_WITH_BA
    return impl_->window->correct(batch);
#else
    (void)batch;
    return false;
#endif
}
void VioEstimator::seed(const VisualMeasurements& batch) {
#ifdef SUPERVO_WITH_BA
    if (impl_->window)
        impl_->window->seed(batch);
#else
    (void)batch;
#endif
}
bool VioEstimator::correctStationary(const cv::Vec3d& w, const cv::Vec3d& a) {
    if (impl_->filter)
        return impl_->filter->correctStationary(w, a);
#ifdef SUPERVO_WITH_BA
    return impl_->window->correctStationary();
#else
    return false;
#endif
}
void VioEstimator::rememberReference(int frame) {
    if (impl_->filter)
        impl_->filter->rememberReference(frame);
#ifdef SUPERVO_WITH_BA
    else
        impl_->window->rememberReference(frame);
#endif
}
void VioEstimator::protectReference(int frame) {
    if (impl_->filter)
        impl_->filter->protectReference(frame);
#ifdef SUPERVO_WITH_BA
    else
        impl_->window->protectReference(frame);
#endif
}
std::optional<cv::Matx44d> VioEstimator::referencePose(int frame) const {
    if (impl_->filter)
        return impl_->filter->referencePose(frame);
#ifdef SUPERVO_WITH_BA
    return impl_->window->referencePose(frame);
#else
    return std::nullopt;
#endif
}
std::pair<double, double> VioEstimator::relativeUncertainty(int frame) const {
    if (impl_->filter)
        return impl_->filter->relativeUncertainty(frame);
#ifdef SUPERVO_WITH_BA
    return impl_->window->relativeUncertainty(frame);
#else
    return {0, 0};
#endif
}
std::vector<LandmarkSeed> VioEstimator::optimizedLandmarks() const {
#ifdef SUPERVO_WITH_BA
    if (impl_->window)
        return impl_->window->optimizedLandmarks();
#endif
    return {};
}
InertialState VioEstimator::state() const {
    if (impl_->filter)
        return impl_->filter->state();
#ifdef SUPERVO_WITH_BA
    return impl_->window->state();
#else
    return {};
#endif
}
EstimationFeedback VioEstimator::feedback() const {
#ifdef SUPERVO_WITH_BA
    if (impl_->window)
        return impl_->window->feedback();
#endif
    return {};
}
WindowDiagnostics VioEstimator::diagnostics() const {
#ifdef SUPERVO_WITH_BA
    if (impl_->window)
        return impl_->window->diagnostics();
#endif
    return {};
}
}
