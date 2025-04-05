cmake_minimum_required(VERSION 3.10)

project(rknn_yolov8_seg_demo)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 启用 Address Sanitizer（可选）
if (ENABLE_ASAN)
    message(STATUS "BUILD WITH ADDRESS SANITIZER")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
    set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

# 添加子目录
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/ utils.out)

# 检测架构
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "64bit")
    set(TARGET_LIB_ARCH lib64)
else()
    message(STATUS "32bit")
    set(TARGET_LIB_ARCH lib)
endif()

# 设置 OpenCV 路径
set(OpenCV_DIR /opt/opencv-4.5.5/opencv_arm64/lib/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS=${OpenCV_LIBS}")

# 设置 FFmpeg 路径
set(FFMPEG_PREFIX "/opt/ffmpeg-4.4.4/tmp")
set(FFMPEG_INCLUDE_DIRS "${FFMPEG_PREFIX}/include")
set(FFMPEG_LIBRARY_DIRS "${FFMPEG_PREFIX}/lib")
set(FFMPEG_LIBRARIES
    "${FFMPEG_LIBRARY_DIRS}/libavformat.so"
    "${FFMPEG_LIBRARY_DIRS}/libavcodec.so"
    "${FFMPEG_LIBRARY_DIRS}/libavutil.so"
    "${FFMPEG_LIBRARY_DIRS}/libswscale.so"
)

# 检查 FFmpeg 库是否存在
foreach(lib ${FFMPEG_LIBRARIES})
    if(NOT EXISTS "${lib}")
        message(FATAL_ERROR "FFmpeg library not found: ${lib}")
    endif()
endforeach()

include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIBRARY_DIRS})
message(STATUS "FFMPEG_INCLUDE_DIRS=${FFMPEG_INCLUDE_DIRS}")
message(STATUS "FFMPEG_LIBRARY_DIRS=${FFMPEG_LIBRARY_DIRS}")
message(STATUS "FFMPEG_LIBRARIES=${FFMPEG_LIBRARIES}")

# 设置 RPATH
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib;${FFMPEG_LIBRARY_DIRS}")

# 源文件
file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

# DMA 和 DRM 头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/dma)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../3rdparty/allocator/drm)

# 根据目标 SOC 选择文件
if (TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    set(postprocess_file rknpu1/postprocess.cc)
    set(yolov8_seg_file rknpu1/yolov8_seg.cc)
else()
    set(postprocess_file rknpu2/postprocess.cc)
    set(yolov8_seg_file rknpu2/yolov8_seg.cc)
endif()

# 添加可执行文件
add_executable(${PROJECT_NAME}
    main.cc
    ${postprocess_file}
    ${yolov8_seg_file}
)

# 启用 OpenMP 支持
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    message(STATUS "OpenMP found: ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found. Please install OpenMP support.")
endif()

# 链接库
if (TARGET_SOC STREQUAL "rk1808" OR TARGET_SOC STREQUAL "rv1109" OR TARGET_SOC STREQUAL "rv1126")
    target_link_libraries(${PROJECT_NAME}
        fileutils
        imageutils
        imagedrawing
        ${OpenCV_LIBS}
        ${FFMPEG_LIBRARIES}
        ${LIBRKNNRT}
    )
else()
    target_link_libraries(${PROJECT_NAME}
        fileutils
        imageutils
        imagedrawing
        ${OpenCV_LIBS}
        ${FFMPEG_LIBRARIES}
        ${LIBRKNNRT}
        ${LIBRGA}
    )
endif()

# Android 平台特定链接
if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(${PROJECT_NAME}
        log
    )
endif()

# Linux 平台特定链接
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(${PROJECT_NAME}
        Threads::Threads
        dl
    )
endif()

# 包含目录
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
)

# 安装规则
install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${LIBRKNNRT} DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/bus.jpg DESTINATION model)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/coco_80_labels_list.txt DESTINATION model)

file(GLOB RKNN_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../model/*.rknn")
install(FILES ${RKNN_FILES} DESTINATION model)