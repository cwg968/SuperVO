#pragma once
#include <frame.h>
#include <sensorTypes.h>
#include <vector>

namespace superVO {
enum class ObservationState { Inlier, Outlier, Removed };
struct ObservationFeedback {
    LandmarkId landmark = 0;
    int frame = -1, camera = 0;
    double pixelError = 0, whitenedSquaredError = 0;
    ObservationState state = ObservationState::Inlier;
};
// Published only after a complete visual update commits. Entries outside the
// optimizer budget are absent, not rejected. Historical entries are diagnostic.
struct EstimationFeedback {
    int frame = -1;
    std::vector<ObservationFeedback> observations;
};
void discardPosteriorOutliers(int frame, const std::vector<LandmarkId>& ids,
                              std::vector<StereoObservation>& pixels,
                              const EstimationFeedback& feedback);
}
