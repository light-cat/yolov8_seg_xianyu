// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "yolov8_seg.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

struct FrameData {
    cv::Mat frame;
    object_detect_result_list od_results;
    bool is_valid;
    FrameData() : is_valid(false) {}
    FrameData(const cv::Mat& f, const object_detect_result_list& res = object_detect_result_list())
        : frame(f.clone()), od_results(res), is_valid(true) {}
};

class ThreadSafeQueue {
private:
    std::queue<FrameData> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    const size_t max_size_ = 30;

public:
    void push(const FrameData& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this]() { return queue_.size() < max_size_; });
        queue_.push(item);
        lock.unlock();
        not_empty_.notify_one();
    }

    FrameData pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this]() { return !queue_.empty(); });
        FrameData item = queue_.front();
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return item;
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
};

void videoCaptureThread(ThreadSafeQueue& input_queue, const std::string& source) {
    cv::VideoCapture cap;
    if (source.empty() || source == "0") {
        cap.open(0);  // Camera
    } else {
        cap.open(source);  // Video file
    }

    if (!cap.isOpened()) {
        printf("Error: Could not open video source %s\n", source.c_str());
        return;
    }

    cv::Mat frame;
    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            break;
        }
        input_queue.push(FrameData(frame));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    input_queue.push(FrameData());  // Signal end
}

void detectionThread(ThreadSafeQueue& input_queue, ThreadSafeQueue& output_queue, rknn_app_context_t& rknn_app_ctx) {
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        FrameData frame_data = input_queue.pop();
        if (!frame_data.is_valid) {
            output_queue.push(FrameData());  // Signal end
            break;
        }

        image_buffer_t img_buf;
        memset(&img_buf, 0, sizeof(image_buffer_t));
        img_buf.width = frame_data.frame.cols;
        img_buf.height = frame_data.frame.rows;
        img_buf.format = IMAGE_FORMAT_RGB888;
        img_buf.size = img_buf.width * img_buf.height * 3;
        img_buf.virt_addr = (unsigned char*)malloc(img_buf.size);
        if (!img_buf.virt_addr) {
            printf("Error: Failed to allocate memory for frame %d\n", frame_count);
            continue;
        }
        memcpy(img_buf.virt_addr, frame_data.frame.data, img_buf.size);

        auto inference_start = std::chrono::steady_clock::now();
        object_detect_result_list od_results;
        if (inference_yolov8_seg_model(&rknn_app_ctx, &img_buf, &od_results) == 0) {
            auto inference_end = std::chrono::steady_clock::now();
            double inference_time_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
            double fps = 1000.0 / inference_time_ms;

            printf("Frame %d: %d objects detected, FPS: %.2f\n", frame_count, od_results.count, fps);

            frame_data.od_results = od_results;
            output_queue.push(frame_data);
        } else {
            printf("Frame %d: Inference failed\n", frame_count);
            output_queue.push(frame_data);
        }

        free(img_buf.virt_addr);
        frame_count++;
    }

    auto total_end_time = std::chrono::steady_clock::now();
    double total_time_s = std::chrono::duration<double>(total_end_time - start_time).count();
    printf("Total frames processed: %d, Average FPS: %.2f\n", frame_count, frame_count / total_time_s);
}

void saveThread(ThreadSafeQueue& output_queue, const std::string& video_source) {
    cv::VideoCapture cap(video_source.empty() || video_source == "0" ? 0 : video_source);
    if (!cap.isOpened()) {
        printf("Error: Could not open video source for properties\n");
        return;
    }
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;  // Fallback FPS if not detected
    cap.release();

    // Prepare video writer with fallback codecs
    cv::VideoWriter writer;
    const char* filename = "output_video.mp4";
    cv::Size size(width, height);

    int codecs[] = {
        cv::VideoWriter::fourcc('H', '2', '6', '4'),  // H.264
        cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),  // XVID
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G')   // MJPEG
    };
    const char* codec_names[] = {"H264", "XVID", "MJPG"};
    int codec_count = sizeof(codecs) / sizeof(codecs[0]);
    bool writer_opened = false;

    for (int i = 0; i < codec_count && !writer_opened; i++) {
        int codec = codecs[i];
        writer.open(filename, codec, fps, size, true);
        if (writer.isOpened()) {
            printf("Using codec: %s\n", codec_names[i]);
            writer_opened = true;
        } else {
            printf("Failed to open video writer with codec %s\n", codec_names[i]);
        }
    }

    if (!writer_opened) {
        printf("Error: Could not open video writer with any codec\n");
        return;
    }

    int frame_count = 0;
    while (true) {
        FrameData frame_data = output_queue.pop();
        if (!frame_data.is_valid) {
            break;
        }

        cv::Mat& frame = frame_data.frame;
        for (int i = 0; i < frame_data.od_results.count; i++) {
            object_detect_result* det_result = &frame_data.od_results.results[i];
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

    rknn_app_context_t rknn_app_ctx{};
    int ret = init_post_process();
    if (ret != 0) {
        printf("init_post_process failed! ret=%d\n", ret);
        return -1;
    }

    ret = init_yolov8_seg_model(model_path, &rknn_app_ctx);
    if (ret != 0) {
        printf("init_yolov8_seg_model failed! ret=%d model_path=%s\n", ret, model_path);
        deinit_post_process();
        return -1;
    }

    {
        ThreadSafeQueue input_queue;
        ThreadSafeQueue output_queue;

        std::thread capture_thread(videoCaptureThread, std::ref(input_queue), std::string(video_source));
        std::thread detect_thread(detectionThread, std::ref(input_queue), std::ref(output_queue), std::ref(rknn_app_ctx));
        std::thread save_thread(saveThread, std::ref(output_queue), std::string(video_source));

        capture_thread.join();
        detect_thread.join();
        save_thread.join();
    }

    deinit_post_process();
    ret = release_yolov8_seg_model(&rknn_app_ctx);
    if (ret != 0) {
        printf("release_yolov8_seg_model failed! ret=%d\n", ret);
        return -1;
    }

    printf("Processing completed.\n");
    return 0;
}