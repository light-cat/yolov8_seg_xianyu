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
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include "yolov8_seg.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "rknn_api.h"

extern "C"
{
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/pixdesc.h>
}

// 添加宏定义控制TCP发送线程
#define ENABLE_TCP_SENDER 0 // 1:启用 0:禁用

// TCP相关配置
#define TCP_PORT 12345
#define TCP_SERVER_IP "192.168.0.140"  // 修改为上位机IP

// 全局模型上下文
static rknn_app_context_t rknn_app_ctx[3];  // 为两个NPU核心准备两个上下文
static bool models_initialized = false;     // 标记模型是否已初始化

// 定义类别颜色表，与示例一致
static unsigned char class_colors[][3] = {
    {255, 56, 56},   // 'FF3838'
    {255, 157, 151}, // 'FF9D97'
    {255, 112, 31},  // 'FF701F'
    {255, 178, 29},  // 'FFB21D'
    {207, 210, 49},  // 'CFD231'
    {72, 249, 10},   // '48F90A'
    {146, 204, 23},  // '92CC17'
    {61, 219, 134},  // '3DDB86'
    {26, 147, 52},   // '1A9334'
    {0, 212, 187},   // '00D4BB'
    {44, 153, 168},  // '2C99A8'
    {0, 194, 255},   // '00C2FF'
    {52, 69, 147},   // '344593'
    {100, 115, 255}, // '6473FF'
    {0, 24, 236},    // '0018EC'
    {132, 56, 255},  // '8438FF'
    {82, 0, 133},    // '520085'
    {203, 56, 255},  // 'CB38FF'
    {255, 149, 200}, // 'FF95C8'
    {255, 55, 199}   // 'FF37C7'
};

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
    const size_t max_size_ = 100;

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

    char size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
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
ThreadSafeQueue<FrameData> input_queue;  // 线程安全的输入队列
ThreadSafeQueue<FrameData> output_queue; // 线程安全的输出队列
ThreadSafeQueue<FrameData> processed_queue; // 新增队列，用于保存处理后的帧
std::atomic<int> frame_counter{0};

// 初始化模型的函数，只在程序启动时调用一次
static int initialize_models(const char* model_path) {
    if (models_initialized) {
        return 0;  // 已经初始化过，直接返回
    }

    for (int i = 0; i < 3; i++) {
        int ret = init_yolov8_seg_model(model_path, &rknn_app_ctx[i]);
        if (ret != 0) {
            printf("init_yolov8_seg_model failed for core %d! ret=%d\n", i, ret);
            for (int j = 0; j < i; j++) {
                release_yolov8_seg_model(&rknn_app_ctx[j]);
            }
            return ret;
        }

        //rknn_core_mask core_mask = (i == 0) ? RKNN_NPU_CORE_0 : RKNN_NPU_CORE_1;
        rknn_core_mask core_mask = RKNN_NPU_CORE_0;
        if(i == 0)
        {
            core_mask = RKNN_NPU_CORE_0;
        }else if(i == 1)
        {
            core_mask = RKNN_NPU_CORE_1;
        }else
        {
            core_mask = RKNN_NPU_CORE_2;
        }

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

#if 0 
void videoCaptureThread(const std::string& source) {
    // FFmpeg 初始化
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    SwsContext* sws_ctx = nullptr;
    int video_stream_idx = -1;

    // 添加优化选项
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "format", "nv12", 0);  // 使用 NV12 格式减少 Ascendingly optimized
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);  // 使用 TCP 传输 RTSP

    // 打开输入
    if (avformat_open_input(&fmt_ctx, source.c_str(), nullptr, &opts) < 0) {
        printf("Error: Could not open video source %s\n", source.c_str());
        running = false;
        return;
    }
    fmt_ctx->probesize = 1024 * 1024;
    fmt_ctx->max_analyze_duration = 5000000;

    // 查找最佳视频流
    video_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        printf("Error: No video stream found\n");
        avformat_close_input(&fmt_ctx);
        running = false;
        return;
    }

    // 配置解码器
    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder_by_name("h264_rkmpp");  // 尝试 RKMPP
    if (!codec) {
        printf("Warning: h264_rkmpp not found, falling back to software decoder\n");
        codec = avcodec_find_decoder(codecpar->codec_id);
    }
    printf("Using decoder: %s\n", codec->name);  // 打印实际使用的解码器名称

    codec_ctx = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx, codecpar);
    
    // 优化解码参数
    codec_ctx->thread_count = 13;
    codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    codec_ctx->thread_type = FF_THREAD_SLICE;
    codec_ctx->skip_frame = AVDISCARD_NONREF;

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        printf("Error: Could not open codec\n");
        avformat_close_input(&fmt_ctx);
        running = false;
        return;
    }

    // 使用缓存上下文进行格式转换
    sws_ctx = sws_getCachedContext(nullptr, 
                                 codec_ctx->width, codec_ctx->height, AV_PIX_FMT_NV12,
                                 1280, 720, AV_PIX_FMT_BGR24,
                                 SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

    // 预分配 Mat
    cv::Mat mat(720, 1280, CV_8UC3);
    uint8_t* out_data[1] = { mat.data };
    int out_linesize[1] = { static_cast<int>(mat.step) };

    // 主循环
    while (running) {
        auto start = std::chrono::steady_clock::now();

        if (av_read_frame(fmt_ctx, packet) < 0) {
            break;
        }

        if (packet->stream_index == video_stream_idx) {
            int ret = avcodec_send_packet(codec_ctx, packet);
            if (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret >= 0) {
                    sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                            out_data, out_linesize);

                    int current_index = frame_counter++;
                    input_queue.push(FrameData(mat.clone(), current_index));
                    
                    auto end = std::chrono::steady_clock::now();
                    double read_time = std::chrono::duration<double, std::milli>(end - start).count();
                    if (frame_counter % 30 == 0) {
                        printf("Frame %d read time: %.2f ms\n", current_index, read_time);
                    }
                }
            }
        }
        av_packet_unref(packet);
    }

    input_queue.push(FrameData());

    // 清理资源
    if (sws_ctx) sws_freeContext(sws_ctx);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
    if (opts) av_dict_free(&opts);
}
#endif




void videoCaptureThread(const std::string& source) {
    // FFmpeg 初始化
    AVFormatContext* fmt_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVFrame* frame = av_frame_alloc();
    AVPacket* packet = av_packet_alloc();
    SwsContext* sws_ctx = nullptr;
    int video_stream_idx = -1;

    // 配置输入选项
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "format", "nv12", 0);

    // 判断输入类型并设置选项
    bool is_usb_camera = (source.find("/dev/video") != std::string::npos);
    if (is_usb_camera) {
        av_dict_set(&opts, "input_format", "mjpeg", 0);
        av_dict_set(&opts, "framerate", "30", 0);
        av_dict_set(&opts, "video_size", "1280x720", 0);
        printf("Input detected as USB camera: %s\n", source.c_str());
    } else {
        printf("Input detected as local video: %s\n", source.c_str());
    }

    // 打开输入源
    int ret = avformat_open_input(&fmt_ctx, source.c_str(), nullptr, &opts);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        printf("Error: Could not open source %s (%s)\n", source.c_str(), errbuf);
        running = false;
        return;
    }
    fmt_ctx->probesize = 1024 * 1024;
    fmt_ctx->max_analyze_duration = 5000000;

    // 查找视频流并识别格式
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        printf("Error: Could not find stream info (%s)\n", errbuf);
        avformat_close_input(&fmt_ctx);
        running = false;
        return;
    }
    
    video_stream_idx = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        printf("Error: No video stream found\n");
        avformat_close_input(&fmt_ctx);
        running = false;
        return;
    }

    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_idx]->codecpar;
    printf("Video Format: %s\n", avcodec_get_name(codecpar->codec_id));
    printf("Resolution: %dx%d\n", codecpar->width, codecpar->height);
    printf("Frame Rate: %.2f fps\n", av_q2d(fmt_ctx->streams[video_stream_idx]->r_frame_rate));
   // printf("Pixel Format: %s\n", av_get_pix_fmt_name(codecpar->format));

    // 配置解码器
    const AVCodec* codec = avcodec_find_decoder_by_name("h264_rkmpp");
    if (!codec) {
        printf("Warning: h264_rkmpp not found, using fallback\n");
        codec = avcodec_find_decoder(codecpar->codec_id);
    }
    printf("Using decoder: %s\n", codec->name);

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx || avcodec_parameters_to_context(codec_ctx, codecpar) < 0) {
        printf("Error: Could not allocate codec context\n");
        avformat_close_input(&fmt_ctx);
        running = false;
        return;
    }
    
    codec_ctx->thread_count = 8;
    codec_ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    codec_ctx->thread_type = FF_THREAD_SLICE;
   // codec_ctx->skip_frame = AVDISCARD_NONREF;
    codec_ctx->skip_frame = AVDISCARD_NONKEY;
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        printf("Error: Could not open codec\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        running = false;
        return;
    }

    // 初始化格式转换
    sws_ctx = sws_getCachedContext(nullptr, 
                                  codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                                  codec_ctx->width, codec_ctx->height, AV_PIX_FMT_BGR24,
                                  SWS_POINT, nullptr, nullptr, nullptr);
    if (!sws_ctx) {
        printf("Error: Could not initialize sws context\n");
        avcodec_free_context(&codec_ctx);
        avformat_close_input(&fmt_ctx);
        running = false;
        return;
    }

    // 预分配 Mat
    static cv::Mat mat(720, 1280, CV_8UC3);
    uint8_t* out_data[1] = { mat.data };
    int out_linesize[1] = { static_cast<int>(mat.step) };

    // 主循环
    while (running) {
        auto t1 = std::chrono::steady_clock::now();
        if (av_read_frame(fmt_ctx, packet) < 0) {
            printf("Warning: End of stream or error\n");
            break;
        }
        auto t2 = std::chrono::steady_clock::now();

        if (packet->stream_index == video_stream_idx) {
            int ret = avcodec_send_packet(codec_ctx, packet);
            if (ret >= 0) {
                ret = avcodec_receive_frame(codec_ctx, frame);
                auto t3 = std::chrono::steady_clock::now();
                if (ret >= 0) {
                    // sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height,
                    //          out_data, out_linesize);

                    // 使用 OpenCV 替代 sws_scale
                    cv::Mat yuv_mat(frame->height * 3 / 2, frame->width, CV_8UC1);
                    memcpy(yuv_mat.data, frame->data[0], frame->width * frame->height); // Y
                    memcpy(yuv_mat.data + frame->width * frame->height, frame->data[1], frame->width * frame->height / 2); // UV
                    cv::cvtColor(yuv_mat, mat, cv::COLOR_YUV2BGR_NV12);

                    auto t4 = std::chrono::steady_clock::now();

                    int current_index = frame_counter++;
                    if (input_queue.size() > 100) {
                        continue; // 丢帧，避免堆积
                    }
                    input_queue.push(FrameData(mat, current_index));
                    auto t5 = std::chrono::steady_clock::now();

                    if (frame_counter % 30 == 0) {
                        double read_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
                        double decode_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
                        double scale_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
                        double queue_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
                        double total_ms = std::chrono::duration<double, std::milli>(t5 - t1).count();
                        printf("Frame %d: Read=%.2f ms, Decode=%.2f ms, Scale=%.2f ms, Queue=%.2f ms, Total=%.2f ms\n",
                               current_index, read_ms, decode_ms, scale_ms, queue_ms, total_ms);
                    }
                }
            }
        }
        av_packet_unref(packet);
    }

    input_queue.push(FrameData()); // 结束信号，空帧

    // 清理资源
    sws_freeContext(sws_ctx);
    avcodec_free_context(&codec_ctx);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avformat_close_input(&fmt_ctx);
    av_dict_free(&opts);
}


void detectionThread(int npu_core, int thread_id, int cpu_id) {
    int ctx_idx = npu_core;  // 每个线程使用对应的模型上下文
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
        printf("Set thread affinity failed\n");
    printf("Bind NPU process %d on CPU %d\n", thread_id, cpu_id);

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

        output_queue.push(frame_data);
    }
}

void saveThread(const std::string& video_source) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
        printf("Set thread affinity failed for save thread\n");
    printf("Bind save thread on CPU 1\n");

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

            // 绘制掩码
            if (ordered_frame.od_results.count >= 1 && ordered_frame.od_results.results_seg[0].seg_mask != nullptr) {
                uint8_t* seg_mask = ordered_frame.od_results.results_seg[0].seg_mask;
                float alpha = 0.5f; // 透明度
                for (int j = 0; j < height; j++) {
                    for (int k = 0; k < width; k++) {
                        int idx = j * width + k;
                        if (seg_mask[idx] != 0) {
                            int cls_idx = seg_mask[idx] % 20;
                            uchar* pixel = frame.ptr<uchar>(j, k);
                            pixel[2] = (uchar)(class_colors[cls_idx][0] * (1 - alpha) + pixel[2] * alpha); // R
                            pixel[1] = (uchar)(class_colors[cls_idx][1] * (1 - alpha) + pixel[1] * alpha); // G
                            pixel[0] = (uchar)(class_colors[cls_idx][2] * (1 - alpha) + pixel[0] * alpha); // B
                        }
                    }
                }
            }

            // 绘制边界框和标签
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

            // 保存视频
           // writer.write(frame);
            frame_count++;

            #if ENABLE_TCP_SENDER
            // 将处理后的帧推送到processed_queue供TCP发送
                processed_queue.push(FrameData(frame, ordered_frame.frame_index, ordered_frame.od_results));
            #endif

            // 释放掩码内存
            if (ordered_frame.od_results.count > 0 && ordered_frame.od_results.results_seg[0].seg_mask != nullptr) {
                free(ordered_frame.od_results.results_seg[0].seg_mask);
                ordered_frame.od_results.results_seg[0].seg_mask = nullptr;
            }

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

#if ENABLE_TCP_SENDER
void tcpSenderThread() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(7, &mask);  // 绑定到CPU核心2
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) < 0)
        printf("Set thread affinity failed for TCP sender thread\n");
    printf("Bind TCP sender thread on CPU 2\n");

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        printf("TCP socket creation failed\n");
        return;
    }

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(TCP_PORT);
    inet_pton(AF_INET, TCP_SERVER_IP, &server_addr.sin_addr);

    if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        printf("TCP connection failed\n");
        close(sock);
        return;
    }

    while (running) {
        FrameData frame_data;
        if (!processed_queue.pop(frame_data) || !frame_data.is_valid) {
            break;
        }

        // 编码图像为JPEG
        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame_data.frame, buffer);
        
        // 发送数据大小
        uint32_t size = buffer.size();
        size = htonl(size);
        if (send(sock, &size, sizeof(size), 0) < 0) {
            printf("Failed to send image size\n");
            break;
        }

        // 发送图像数据
        if (send(sock, buffer.data(), buffer.size(), 0) < 0) {
            printf("Failed to send image data\n");
            break;
        }

        printf("Sent processed frame %d to TCP server, size: %u bytes\n", 
               frame_data.frame_index, (unsigned int)buffer.size());
    }

    close(sock);
    printf("TCP sender thread terminated\n");
}
#endif

int main(int argc, char **argv) {
    if (argc != 3) {
        printf("%s <model_path> <video_path_or_camera_index>\n", argv[0]);
        printf("Use '0' for camera, or specify a video file path\n");
        return -1;
    }

    const char *model_path = argv[1];
    const char *video_source = argv[2];

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
    capture_thread.detach();

    std::thread detect_thread1(detectionThread, 0, 1, 4);
    setThreadAffinity(detect_thread1, 4);
    detect_thread1.detach();

    std::thread detect_thread2(detectionThread, 1, 2, 5);
    setThreadAffinity(detect_thread2, 5);
    detect_thread2.detach();

    std::thread detect_thread3(detectionThread, 2, 3, 6);
    setThreadAffinity(detect_thread3, 6);
    detect_thread3.detach();

    std::thread save_thread(saveThread, std::string(video_source));
    setThreadAffinity(save_thread, 1);
    save_thread.detach();

#if ENABLE_TCP_SENDER
    std::thread tcp_thread(tcpSenderThread);
    setThreadAffinity(tcp_thread, 7);
    tcp_thread.detach();
#endif

    printf("Press 'q' and Enter to quit...\n");
    char input;
    while (true) {
        input = getchar();
        if (input == 'q' || input == 'Q') {
            running = false;
            break;
        }
    }

    while (!input_queue.empty() || !output_queue.empty() || !processed_queue.empty()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    cleanup_models();
    deinit_post_process();
    printf("Processing completed.\n");
    return 0;
}