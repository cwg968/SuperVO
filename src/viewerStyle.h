#pragma once
#include "viewerScene.h"
#include <pangolin/display/default_font.h>
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <algorithm>
#include <array>
#include <iomanip>
#include <sstream>

namespace superVO {
namespace viewerStyle {
inline void rectangle(float x, float y, float w, float h, float r, float g, float b) {
    glColor3f(r, g, b);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y + h);
    glVertex2f(x, y + h);
    glEnd();
}
inline void text(float x, float y, const std::string& value, float scale = 1,
                 cv::Vec3f color = {.82f, .87f, .92f}) {
    glColor3fv(color.val);
    glPushMatrix();
    glTranslatef(x, y, 0);
    glScalef(scale, scale, 1);
    pangolin::default_font().Text(value).Draw();
    glPopMatrix();
}
inline std::string fixed(double value, int precision = 1) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}
// Robust display framing only. Outlying triangulations must not set the zoom level.
inline std::pair<cv::Vec3d, double> bounds(const ViewerScene& scene) {
    cv::Vec3d low(-1, -1, -1), high(1, 1, 1);
    if (!scene.landmarks.empty()) {
        std::array<std::vector<double>, 3> axes;
        const size_t stride = std::max<size_t>(1, scene.landmarks.size() / 2048);
        size_t index = 0;
        for (const auto& [id, p] : scene.landmarks) {
            (void)id;
            if (index++ % stride)
                continue;
            axes[0].push_back(p.x);
            axes[1].push_back(p.y);
            axes[2].push_back(p.z);
        }
        for (int i = 0; i < 3; ++i) {
            auto& values = axes[i];
            std::sort(values.begin(), values.end());
            low[i] = values[values.size() / 20];
            high[i] = values[(values.size() - 1) * 19 / 20];
        }
    }
    for (const auto& p : scene.trajectory)
        for (int i = 0; i < 3; ++i) {
            low[i] = std::min(low[i], p[i]);
            high[i] = std::max(high[i], p[i]);
        }
    return {(low + high) * .5, std::max(1.5, cv::norm(high - low) * .55)};
}
inline void hud(const ViewerFrame* frame, const ViewerScene& scene, bool presentation,
                bool overview, bool orbit, bool finished, double elapsed) {
    const auto& screen = pangolin::DisplayBase();
    screen.ActivatePixelOrthographic();
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    const float left = presentation ? 0 : 190;
    const float width = screen.v.w - left, height = screen.v.h;
    rectangle(left, height - 82, width, 82, .035f, .05f, .07f);
    rectangle(left + 22, height - 52, 4, 30, .18f, .94f, .65f);
    text(left + 38, height - 44, "SUPERVO / VIO", 1.65f, {.94f, .97f, 1.f});
    text(left + 38, height - 65, "FEATURE TRACKING  /  SPARSE RECONSTRUCTION", .8f,
         {.43f, .58f, .68f});
    const std::string state = finished ? "COMPLETE" : frame ? frame->status : "STARTING";
    const cv::Vec3f stateColor =
        frame && (frame->status == "LOST" || frame->status == "INIT FAILED")
            ? cv::Vec3f(1.f, .36f, .38f)
            : cv::Vec3f(.25f, .94f, .65f);
    text(left + width - 224, height - 39, state, 1.1f, stateColor);
    text(left + width - 224, height - 62,
         fixed(elapsed) + " s  /  FRAME " + std::to_string(frame ? frame->frameIndex : 0), .85f);
    rectangle(left, 0, width, 64, .035f, .05f, .07f);
    const auto features = frame && frame->geometry.timestampNs == frame->timestampNs
                              ? frame->geometry.measuredLeft.size()
                              : 0;
    text(left + 24, 36, std::to_string(features) + "  MEASURED FEATURES", .9f, {.25f, .94f, .65f});
    text(left + width * .35f, 36, std::to_string(scene.landmarks.size()) + "  MAP POINTS", .9f,
         {.3f, .76f, 1.f});
    text(left + width * .7f, 36, "PROCESS  " + fixed(frame ? frame->processingMs : 0) + " ms", .9f);
    text(left + 24, 14, "P  presentation   F  follow   O  overview   R  orbit   SPACE  pause", .75f,
         {.43f, .58f, .68f});
    text(left + width - 150, 14,
         orbit      ? "ORBIT VIEW"
         : overview ? "MAP OVERVIEW"
                    : "CAMERA VIEW",
         .75f, {.43f, .58f, .68f});
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}
}
}
