./build-linux.sh -t rk3588 -a aarch64 -d yolov8_seg

cd /3588_test/yolov8_seg/rknn_model_zoo/install/rk3588_linux_aarch64/rknn_yolov8_seg_demo
scp rknn_yolov8_seg_demo teamhd@192.168.0.111:/home/teamhd/RKNN/data/rk3588_linux_aarch64/rknn_yolov8_seg_demo
cd ../../../