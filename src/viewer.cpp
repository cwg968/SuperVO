#include <viewer.h>
#include "viewerScene.h"
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/display/widgets.h>
#include <pangolin/gl/gl.h>
#include <pangolin/handler/handler.h>
#include <pangolin/var/var.h>
#include <opencv2/imgproc.hpp>
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace superVO {
namespace {
constexpr const char* windowTitle = "SuperVIO | Left features & sparse 3D map";
pangolin::OpenGlMatrix glPose(const cv::Matx44d& pose) {
    pangolin::OpenGlMatrix result;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            result.m[c * 4 + r] = pose(r, c);
    return result;
}
void cameraFrustum(const cv::Matx44d& pose, const cv::Matx33d& K, cv::Size size, double scale) {
    const auto transform = glPose(pose);
    glPushMatrix();
    glMultMatrixd(transform.m);
    const std::array<cv::Vec3d, 4> corners = {
        cv::Vec3d(-K(0, 2) / K(0, 0), -K(1, 2) / K(1, 1), 1) * scale,
        cv::Vec3d((size.width - K(0, 2)) / K(0, 0), -K(1, 2) / K(1, 1), 1) * scale,
        cv::Vec3d((size.width - K(0, 2)) / K(0, 0), (size.height - K(1, 2)) / K(1, 1), 1) * scale,
        cv::Vec3d(-K(0, 2) / K(0, 0), (size.height - K(1, 2)) / K(1, 1), 1) * scale};
    glBegin(GL_LINES);
    for (size_t k = 0; k < corners.size(); ++k) {
        glVertex3d(0, 0, 0);
        glVertex3dv(corners[k].val);
        glVertex3dv(corners[k].val);
        glVertex3dv(corners[(k + 1) % corners.size()].val);
    }
    glEnd();
    glPopMatrix();
}
cv::Mat featureImage(const ViewerFrame& frame) {
    cv::Mat image;
    cv::cvtColor(frame.leftImage, image, cv::COLOR_GRAY2RGB);
    auto draw = [&](const auto& points, cv::Scalar color, int radius) {
        for (const auto& p : points)
            if (std::isfinite(p.x) && std::isfinite(p.y) && p.x >= 0 && p.y >= 0 &&
                p.x < image.cols && p.y < image.rows)
                cv::circle(image, p, radius, color, 1, cv::LINE_AA);
    };
    if (frame.geometry.timestampNs == frame.timestampNs) {
        draw(frame.geometry.measuredLeft, {40, 240, 120}, 2);
    }
    cv::rectangle(image, {0, 0}, {image.cols, 46}, {10, 16, 22}, cv::FILLED);
    cv::putText(image, "LEFT / " + frame.status + " / frame " + std::to_string(frame.frameIndex),
                {10, 18}, cv::FONT_HERSHEY_SIMPLEX, .45, {225, 235, 245}, 1, cv::LINE_AA);
    cv::putText(image, "green: current measured features", {10, 37}, cv::FONT_HERSHEY_SIMPLEX, .4,
                {190, 205, 220}, 1, cv::LINE_AA);
    return image;
}
}

struct Viewer::Impl {
    cv::Matx33d K;
    cv::Size imageSize;
    std::mutex mutex;
    std::condition_variable changed;
    std::shared_ptr<const ViewerFrame> pending;
    std::exception_ptr error;
    std::atomic<bool> stop{false}, open{false}, paused{false}, finished{false};
    uint64_t submitted = 0, rendered = 0;
    std::thread thread;

    Impl(const cv::Mat& intrinsics, cv::Size size) : imageSize(size) {
        if (intrinsics.type() != CV_64F || intrinsics.rows != 3 || intrinsics.cols != 3 ||
            !cv::checkRange(intrinsics) || size.width <= 0 || size.height <= 0 ||
            intrinsics.at<double>(0, 0) <= 0 || intrinsics.at<double>(1, 1) <= 0)
            throw std::invalid_argument("Invalid viewer camera");
        K = intrinsics;
        std::promise<void> ready;
        auto started = ready.get_future();
        thread = std::thread([this, ready = std::move(ready)]() mutable { run(ready); });
        try {
            started.get();
        } catch (...) {
            thread.join();
            throw;
        }
    }
    ~Impl() {
        stop = true;
        changed.notify_all();
        if (thread.joinable())
            thread.join();
    }
    void run(std::promise<void>& ready) {
        bool created = false, announced = false;
        try {
            pangolin::CreateWindowAndBind(windowTitle, 1440, 860);
            created = true;
            {
                glEnable(GL_DEPTH_TEST);
                glClearColor(.055f, .07f, .09f, 1.f);
                pangolin::CreatePanel("ui").SetBounds(0, 1, 0, pangolin::Attach::Pix(190));
                pangolin::Var<bool> follow("ui.Follow camera", true, true);
                pangolin::Var<bool> showPoints("ui.Map points", true, true);
                pangolin::Var<bool> showKeys("ui.Keyframes", true, true);
                pangolin::Var<bool> showPath("ui.Trajectory", true, true);
                pangolin::Var<bool> pause("ui.Pause", false, true);
                pangolin::Var<bool> reset("ui.Reset view", false, false);
                pangolin::Var<std::string> status("ui.Status", "Starting");
                pangolin::Var<int> frameNumber("ui.Frame", 0), pointCount("ui.Map size", 0);
                pangolin::Var<int> inliers("ui.Inliers", 0);
                pangolin::Var<double> trackingMs("ui.Process ms", 0.);
                const auto home = pangolin::ModelViewLookAt(0, -3, -8, 0, 0, 1, pangolin::AxisNegY);
                pangolin::OpenGlRenderState camera(
                    pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, .05, 1000), home);
                pangolin::Handler3D handler(camera);
                auto& imageView =
                    pangolin::CreateDisplay().SetBounds(0, 1, pangolin::Attach::Pix(190), .51,
                                                        double(imageSize.width) / imageSize.height);
                auto& mapView =
                    pangolin::CreateDisplay().SetBounds(0, 1, .51, 1).SetHandler(&handler);
                pangolin::GlTexture texture(imageSize.width, imageSize.height, GL_RGB, false, 0,
                                            GL_RGB, GL_UNSIGNED_BYTE);
                ViewerScene scene;
                std::shared_ptr<const ViewerFrame> frame;
                bool hasImage = false;
                open = true;
                ready.set_value();
                announced = true;
                while (!stop && !pangolin::ShouldQuit()) {
                    const auto tick = std::chrono::steady_clock::now();
                    uint64_t serial = 0;
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (pending) {
                            frame = std::move(pending);
                            serial = submitted;
                        }
                    }
                    if (serial) {
                        scene.update(*frame);
                        const auto image = featureImage(*frame);
                        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
                        texture.Upload(image.data, GL_RGB, GL_UNSIGNED_BYTE);
                        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
                        hasImage = true;
                        frameNumber = frame->frameIndex;
                        inliers = frame->inliers;
                        trackingMs = frame->processingMs;
                        pointCount = int(scene.landmarks.size());
                    }
                    status = finished ? "Finished" : frame ? frame->status : "Starting";
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        paused = bool(pause);
                    }
                    changed.notify_all();
                    if (bool(reset)) {
                        camera.SetModelViewMatrix(home);
                        follow = true;
                        reset = false;
                    }
                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                    mapView.Activate(camera);
                    if (bool(follow) && frame && frame->T_world_rect) {
                        camera.Follow(glPose(*frame->T_world_rect));
                        mapView.Activate(camera);
                    }
                    glLineWidth(1.f);
                    glBegin(GL_LINES);
                    // These axes define the odometry frame, not an estimated ground plane.
                    glColor3f(1, .2f, .2f);
                    glVertex3d(0, 0, 0);
                    glVertex3d(1, 0, 0);
                    glColor3f(.2f, 1, .2f);
                    glVertex3d(0, 0, 0);
                    glVertex3d(0, 1, 0);
                    glColor3f(.2f, .4f, 1);
                    glVertex3d(0, 0, 0);
                    glVertex3d(0, 0, 1);
                    glEnd();
                    if (bool(showPoints)) {
                        glPointSize(2.f);
                        glColor3f(.48f, .58f, .67f);
                        glBegin(GL_POINTS);
                        for (const auto& [id, p] : scene.landmarks) {
                            (void)id;
                            glVertex3d(p.x, p.y, p.z);
                        }
                        glEnd();
                        if (frame && frame->geometry.timestampNs == frame->timestampNs) {
                            glColor3f(.15f, .95f, .5f);
                            glPointSize(3.f);
                            glBegin(GL_POINTS);
                            for (const auto& p : frame->geometry.landmarks)
                                glVertex3d(p.world.x, p.world.y, p.world.z);
                            glEnd();
                        }
                    }
                    if (bool(showPath)) {
                        glLineWidth(2.f);
                        glColor3f(1.f, .65f, .2f);
                        glBegin(GL_LINE_STRIP);
                        for (const auto& p : scene.trajectory)
                            glVertex3dv(p.val);
                        glEnd();
                    }
                    if (bool(showKeys)) {
                        glLineWidth(1.f);
                        glColor3f(.3f, .45f, .6f);
                        for (const auto& pose : scene.keyframes)
                            cameraFrustum(pose, K, imageSize, .12);
                    }
                    if (frame && frame->T_world_rect) {
                        glLineWidth(2.f);
                        glColor3f(.15f, .8f, 1.f);
                        cameraFrustum(*frame->T_world_rect, K, imageSize, .3);
                    }
                    if (hasImage) {
                        glDisable(GL_DEPTH_TEST);
                        imageView.Activate();
                        glColor3f(1, 1, 1);
                        texture.RenderToViewportFlipY();
                        glEnable(GL_DEPTH_TEST);
                    }
                    pangolin::FinishFrame();
                    if (serial) {
                        std::lock_guard<std::mutex> lock(mutex);
                        rendered = serial;
                        changed.notify_all();
                    }
                    std::this_thread::sleep_until(tick + std::chrono::milliseconds(33));
                }
            }
            pangolin::DestroyWindow(windowTitle);
            created = false;
        } catch (...) {
            const auto failure = std::current_exception();
            {
                std::lock_guard<std::mutex> lock(mutex);
                error = failure;
            }
            if (!announced)
                ready.set_exception(failure);
            if (created) {
                try {
                    pangolin::DestroyWindow(windowTitle);
                } catch (...) {
                    // Preserve the original startup/rendering error for the producer.
                }
            }
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            open = false;
        }
        changed.notify_all();
    }
};

Viewer::Viewer(const cv::Mat& K, cv::Size size) : impl_(std::make_unique<Impl>(K, size)) {}
Viewer::~Viewer() = default;
void Viewer::submit(ViewerFrame frame) {
    if (frame.leftImage.type() != CV_8UC1 || frame.leftImage.size() != impl_->imageSize)
        throw std::invalid_argument("Viewer requires a current rectified grayscale left image");
    frame.leftImage = frame.leftImage.clone();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->pending = std::make_shared<const ViewerFrame>(std::move(frame));
    ++impl_->submitted;
}
bool Viewer::waitUntilRunning() {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->changed.wait(lock, [&] { return !impl_->open || !impl_->paused; });
    if (impl_->error)
        std::rethrow_exception(impl_->error);
    return impl_->open;
}
void Viewer::finish(bool waitForClose) {
    impl_->finished = true;
    if (!waitForClose) {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->changed.wait_for(lock, std::chrono::seconds(2), [&] {
            return !impl_->open || impl_->rendered == impl_->submitted;
        });
        impl_->stop = true;
    }
    if (impl_->thread.joinable())
        impl_->thread.join();
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->error)
        std::rethrow_exception(impl_->error);
}
}
