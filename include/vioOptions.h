#pragma once
#include <visualMeasurements.h>
namespace superVO {
enum class VioBackend { Eskf, Window };
struct WindowOptions {
    int states = 10, landmarks = 300, iterations = 6, minInliers = 20;
    bool observationFeedback = false;
    bool resolveAfterPruning = false;
    int auditEvery = 0; // Read-only snapshots; zero disables diagnostics.
};
struct VioOptions {
    VioBackend backend = VioBackend::Eskf;
    ObservationNoise noise = ObservationNoise::Patch;
    WindowOptions window;
};
}
