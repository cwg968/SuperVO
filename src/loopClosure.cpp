#include <loopClosure.h>
#include <opencv2/calib3d.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <exception>
#include <chrono>
#include <algorithm>
#include <cmath>
namespace superVO {
namespace {
double positionNorm(const cv::Matx44d& t) {
    return cv::norm(cv::Vec3d(t(0, 3), t(1, 3), t(2, 3)));
}
double rotationNorm(const cv::Matx44d& t) {
    cv::Vec3d a;
    cv::Rodrigues(cv::Mat(t).rowRange(0, 3).colRange(0, 3), a);
    return cv::norm(a);
}
}
struct LoopClosure::Impl {
    StereoCalibration camera;
    LoopOptions options;
    mutable std::mutex mutex;
    std::condition_variable ready, space;
    std::deque<LoopKeyframe> queue;
    bool stopping = false;
    std::thread worker;
    std::exception_ptr error;
    LoopReport report;
    LoopCorrection published;
    std::unique_ptr<GlobalPoseGraph> graph;
    std::vector<cv::Mat> descriptors;
    struct Tentative {
        TimestampNs time = 0, referenceTime = 0;
        cv::Matx44d correction;
    };
    std::optional<Tentative> previous;
    TimestampNs lastLoop = -1;
    Impl(StereoCalibration c, LoopOptions o) : camera(std::move(c)), options(o) {
        camera.intrinsics = camera.intrinsics.clone();
        if (options.keyframeSeconds <= 0 || options.exclusionSeconds <= options.keyframeSeconds ||
            options.candidates < 1 || options.minInliers < 20 || options.pixelThreshold <= 0)
            throw std::invalid_argument("Invalid loop options");
        if (!options.synchronous)
            worker = std::thread([this] { run(); });
    }
    void process(LoopKeyframe key) {
        if (!report.keyframes.empty() && key.timestamp <= report.keyframes.back().timestamp)
            throw std::invalid_argument("Loop timestamps must increase");
        if (!report.keyframes.empty() &&
            (key.timestamp - report.keyframes.back().timestamp) * 1e-9 < options.keyframeSeconds)
            return;
        auto start = std::chrono::steady_clock::now();
        auto pooled = pooledSuperPoint(key.features->descriptors);
        if (!graph)
            graph = std::make_unique<GlobalPoseGraph>(key.gravity);
        int index = int(report.keyframes.size());
        graph->addNode(key.odomPose);
        std::vector<std::pair<double, int>> candidates;
        for (int i = 0; i < index; ++i)
            if ((key.timestamp - report.keyframes[i].timestamp) * 1e-9 >= options.exclusionSeconds)
                candidates.emplace_back(pooled.dot(descriptors[i]), i);
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](auto a, auto b) { return a.first > b.first; });
        if (candidates.size() > size_t(options.candidates))
            candidates.resize(options.candidates);
        report.keyframes.push_back(key);
        descriptors.push_back(pooled);
        for (auto [score, id] : candidates) {
            LoopTrial trial;
            trial.current = index;
            trial.reference = id;
            trial.similarity = score;
            trial.geometry = verifyLoop(report.keyframes[id], key, camera, options);
            if (trial.geometry.valid) {
                const auto& reference = report.keyframes[id];
                auto transform = reference.odomPose *
                                 trial.geometry.currentFromReference.inv(cv::DECOMP_SVD) *
                                 key.odomPose.inv(cv::DECOMP_SVD);
                trial.decision = 1;
                bool agrees = false;
                if (previous && (key.timestamp - previous->time) * 1e-9 <= 4 &&
                    key.timestamp > previous->time &&
                    std::abs(double(reference.timestamp - previous->referenceTime)) * 1e-9 <= 6) {
                    auto delta = previous->correction.inv(cv::DECOMP_SVD) * transform;
                    agrees = positionNorm(delta) < .4 && rotationNorm(delta) < .10;
                    if (!agrees)
                        trial.decision = 2;
                }
                if (agrees && (lastLoop < 0 || (key.timestamp - lastLoop) * 1e-9 >= 5)) {
                    auto measured = trial.geometry.currentFromReference.inv(cv::DECOMP_SVD);
                    bool accepted = graph->addLoop(id, index, measured);
                    trial.decision = accepted ? 4 : 3;
                    if (accepted) {
                        lastLoop = key.timestamp;
                        auto poses = graph->poses();
                        std::lock_guard<std::mutex> lock(mutex);
                        ++published.version;
                        published.anchor = key.timestamp;
                        published.mapFromOdom = poses.back() * key.odomPose.inv(cv::DECOMP_SVD);
                    }
                }
                previous = Tentative{key.timestamp, reference.timestamp, transform};
            }
            trial.milliseconds =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count();
            report.trials.push_back(trial);
            if (trial.geometry.valid)
                break; // At most one verified place per query.
        }
    }
    void run() {
        cv::setRNGSeed(0);
        try {
            for (;;) {
                LoopKeyframe item;
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    ready.wait(lock, [&] { return stopping || !queue.empty(); });
                    if (queue.empty() && stopping)
                        break;
                    item = std::move(queue.front());
                    queue.pop_front();
                    space.notify_one();
                }
                process(std::move(item));
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            error = std::current_exception();
            stopping = true;
            space.notify_all();
        }
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        space.notify_all();
        if (worker.joinable())
            worker.join();
    }
};
LoopClosure::LoopClosure(StereoCalibration c, LoopOptions o)
    : impl_(std::make_unique<Impl>(std::move(c), o)) {}
LoopClosure::~LoopClosure() {
    impl_->stop();
}
void LoopClosure::submit(LoopKeyframe key) {
    if (!key.features)
        throw std::invalid_argument("Loop keyframe missing descriptors");
    if (impl_->options.synchronous) {
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            if (impl_->stopping)
                throw std::logic_error("Loop worker stopped");
        }
        auto saved = cv::theRNG().state;
        try {
            cv::setRNGSeed(0);
            impl_->process(std::move(key));
        } catch (...) {
            cv::theRNG().state = saved;
            throw;
        }
        cv::theRNG().state = saved;
        return;
    }
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->space.wait(lock, [&] { return impl_->stopping || impl_->queue.size() < 16; });
    if (impl_->error)
        std::rethrow_exception(impl_->error);
    if (impl_->stopping)
        throw std::logic_error("Loop worker stopped");
    impl_->queue.push_back(std::move(key));
    impl_->report.queuePeak = std::max(impl_->report.queuePeak, impl_->queue.size());
    impl_->ready.notify_one();
}
LoopCorrection LoopClosure::correction() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->published;
}
LoopReport LoopClosure::finish() {
    impl_->stop();
    if (impl_->error)
        std::rethrow_exception(impl_->error);
    impl_->report.globalPoses = impl_->graph ? impl_->graph->poses() : std::vector<cv::Matx44d>{};
    impl_->report.edges = impl_->graph ? impl_->graph->edges() : std::vector<GraphEdge>{};
    impl_->report.version = impl_->published.version;
    return impl_->report;
}
}
