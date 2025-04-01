#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <map>
#include <sched.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "yolov8_seg.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "rknn_api.h"

// 全局模型上下文
static rknn_app_context_t rknn_app_ctx[2];  // 为两个NPU核心准备两个上下文
static std::mutex model_mutex[2];          // 保护每个模型的互斥锁
static bool models_initialized = false;    // 标记模型是否已初始化

struct FrameData {
    cv::Mat frame;
    object_detect_result_list od_results;
    int frame_index;
    bool is_valid;
    FrameData() : frame_index(-1), is_valid(false) {}
    FrameData(const cv::Mat& f, int idx, const object_detect_result_list& res = object_detect_result_list())
        : frame(f.clone()), od_results(res), frame_index(idx), is_valid(true) {}
};

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    const size_t max_size_ = 30;

public:
    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() { return queue_.size() < max_size_; });
        queue_.push(item);
        lock.unlock();
        not_empty_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return !queue_.empty(); });
        item = queue_.front();
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return true;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

void setThreadAffinity(std::thread& t, int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    int ret = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (ret != 0) {
        printf("Warning: Failed to set affinity to CPU core %d, error: %d\n", core_id, ret);
    }
}

std::atomic<bool> running{true};
ThreadSafeQueue<FrameData> input_queue;
ThreadSafeQueue<FrameData> output_queue;
std::atomic<int> frame_counter{0};

// 初始化模型的函数，只在程序启动时调用一次
static int initialize_models(const char* model_path) {
    if (models_initialized) {
        return 0;  // 已经初始化过，直接返回
    }

    for (int i = 0; i < 2; i++) {
        int ret = init_yolov8_seg_model(model_path, &rknn_app_ctx[i]);
        if (ret != 0) {
            printf("init_yolov8_seg_model failed for core %d! ret=%d\n", i, ret);
            for (int j = 0; j < i; j++) {
                release_yolov8_seg_model(&rknn_app_ctx[j]);
            }
            return ret;
        }

        rknn_core_mask core_mask = (i == 0) ? RKNN_NPU_CORE_0 : RKNN_NPU_CORE_1;
        ret = rknn_set_core_mask(rknn_app_ctx[i].rknn_ctx, core_mask);
        if (ret != 0) {
            printf("Failed to set NPU core mask to %d, ret=%d\n", i, ret);
            for (int j = 0; j <= i; j++) {
                release_yolov8_seg_model(&rknn_app_ctx[j]);
            }
            return ret;
        }
    }
    models_initialized = true;
    printf("Models initialized successfully for both NPU cores.\n");
    return 0;
}

// 清理模型的函数，只在程序结束时调用一次
static void cleanup_models() {
    if (models_initialized) {
        for (int i = 0; i < 2; i++) {
            release_yolov8_seg_model(&rknn_app_ctx[i]);
        }
        models_initialized = false;
        printf("Models cleaned up.\n");
    }
}

void videoCaptureThread(const std::string& source) {
    cv::VideoCapture cap;
    if (source.empty() || source == "0") {
        cap.open(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FPS, 30);
    } else {
        cap.open(source);
    }

    if (!cap.isOpened()) {
        printf("Error: Could not open video source %s\n", source.c_str());
        running = false;
        return;
    }

    cv::Mat frame;
    while (running) {
        if (!cap.read(frame) || frame.empty()) {
            printf("Warning: Failed to read frame\n");
            break;
        }
        int current_index = frame_counter++;
        input_queue.push(FrameData(frame, current_index));
    }
    input_queue.push(FrameData());
    cap.release();
}



void detectionThread(int npu_core, int thread_id, int cpu_id) {
    int ctx_idx = npu_core;  // 每个线程使用对应的模型上下文
    cpu_set_t mask;

	CPU_ZERO(&mask);
	CPU_SET(cpu_id, &mask);

	if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
		printf("set thread affinity failed\n");

	printf("Bind NPU process on CPU %d\n", cpu_id);

    if (!models_initialized) {
        printf("Thread %d: Models not initialized, exiting.\n", thread_id);
        output_queue.push(FrameData());
        return;
    }

    while (running) {
        FrameData frame_data;
        if (!input_queue.pop(frame_data) || !frame_data.is_valid) {
            output_queue.push(FrameData());
            break;
        }

        image_buffer_t img;
        img.width = frame_data.frame.cols;
        img.height = frame_data.frame.rows;
        img.format = IMAGE_FORMAT_RGB888;
        img.size = img.width * img.height * 3;
        img.virt_addr = frame_data.frame.data;

        {
            std::lock_guard<std::mutex> lock(model_mutex[ctx_idx]);
            auto start = std::chrono::steady_clock::now();
            object_detect_result_list od_results;
            if (inference_yolov8_seg_model(&rknn_app_ctx[ctx_idx], &img, &od_results) == 0) {
                auto end = std::chrono::steady_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                printf("Thread %d (NPU Core %d): Detected %d objects in %.2f ms for frame %d\n", 
                       thread_id, npu_core, od_results.count, time_ms, frame_data.frame_index);
                frame_data.od_results = od_results;
            } else {
                printf("Thread %d (NPU Core %d): Inference failed for frame %d\n", 
                       thread_id, npu_core, frame_data.frame_index);
                frame_data.od_results.count = 0;
            }
        }

        output_queue.push(frame_data);
    }
}

void saveThread(const std::string& video_source) {
    cv::VideoCapture cap(video_source.empty() || video_source == "0" ? 0 : video_source);
    if (!cap.isOpened()) {
        printf("Error: Could not open video source for properties\n");
        return;
    }
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;
    cap.release();

    cv::VideoWriter writer;
    const char* filename = "output_video.mp4";
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    writer.open(filename, codec, fps, cv::Size(width, height), true);
    if (!writer.isOpened()) {
        printf("Error: Could not open video writer\n");
        return;
    }

    std::map<int, FrameData> frame_buffer;
    const int MAX_BUFFER_SIZE = 100;
    int expected_index = 0;
    int frame_count = 0;
    bool end_signal_received = false;

    while (running || !output_queue.empty() || !frame_buffer.empty()) {
        FrameData frame_data;
        if (output_queue.pop(frame_data)) {
            if (!frame_data.is_valid) {
                end_signal_received = true;
            } else {
                frame_buffer[frame_data.frame_index] = frame_data;
                if (frame_buffer.size() > MAX_BUFFER_SIZE) {
                    int oldest_index = frame_buffer.begin()->first;
                    if (oldest_index < expected_index) {
                        printf("Warning: Buffer full, dropping frame %d\n", oldest_index);
                        frame_buffer.erase(oldest_index);
                    }
                }
            }
        }

        while (frame_buffer.find(expected_index) != frame_buffer.end()) {
            FrameData& ordered_frame = frame_buffer[expected_index];
            cv::Mat& frame = ordered_frame.frame;

            for (int i = 0; i < ordered_frame.od_results.count; i++) {
                object_detect_result* det_result = &ordered_frame.od_results.results[i];
                int x1 = det_result->box.left;
                int y1 = det_result->box.top;
                int x2 = det_result->box.right;
                int y2 = det_result->box.bottom;
                int cls_id = det_result->cls_id;

                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 255), 3);
                char text[256];
                snprintf(text, sizeof(text), "%s %.1f%%", coco_cls_to_name(cls_id), det_result->prop * 100);
                cv::putText(frame, text, cv::Point(x1, y1 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }

            writer.write(frame);
            frame_count++;
            frame_buffer.erase(expected_index);
            expected_index++;
        }

        if (end_signal_received && frame_buffer.empty()) {
            break;
        }
    }

    writer.release();
    printf("Video saved as '%s' with %d frames\n", filename, frame_count);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_path> <video_path_or_camera_index>\n", argv[0]);
        printf("Use '0' for camera, or specify a video file path\n");
        return -1;
    }

    const char *model_path = argv[1];
    const char *video_source = argv[2];

    // 一次性初始化两个模型
    if (initialize_models(model_path) != 0) {
        printf("Failed to initialize models, exiting.\n");
        return -1;
    }

    int ret = init_post_process();
    if (ret != 0) {
        printf("init_post_process failed! ret=%d\n", ret);
        cleanup_models();
        return -1;
    }

    std::thread capture_thread(videoCaptureThread, std::string(video_source));
    setThreadAffinity(capture_thread, 0);

    std::thread detect_thread1(detectionThread, 0, 1, 4);
    std::thread detect_thread2(detectionThread, 1, 2, 5);

    std::thread save_thread(saveThread, std::string(video_source));
    setThreadAffinity(save_thread, 1);

    capture_thread.join();
    detect_thread1.join();
    detect_thread2.join();
    save_thread.join();

    // 程序结束时清理模型
    cleanup_models();
    deinit_post_process();
    printf("Processing completed.\n");
    return 0;
}