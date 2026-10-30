# 数据输入

## EuRoC

`--format euroc` 或自动检测接受序列根目录或 `mav0` 目录。
输入包括 `cam0/data`、`cam1/data`、时间戳清单和各自的 `sensor.yaml`。
仅配对整数纳秒时间戳完全相同的左右图像，并根据标定做双目校正。
VIO 还需 `imu0/data.csv`、`imu0/sensor.yaml` 及相机与 body 外参。

## KITTI

`--format kitti` 读取序列的 `calib.txt`、`times.txt` 和双目图像。
`--kitti-cameras auto|gray|color` 选择 `image_0/image_1` 或 `image_2/image_3`；
自动选择优先灰度。彩色使用 P2/P3 标定与完整平移做双目校正。
KITTI 当前适配器不提供 IMU。

```sh
build-ba/superVO --dataset /path/to/KITTI/sequences/00 --format kitti \
  --kitti-cameras color --image-scale 0.5 --backend tensorrt \
  --engine /path/to/kitti-624x192.engine --output results/kitti00
```

缩放按实际像素中心更新内参，米制基线不变。
例如 1241×376 缩至 620×188 后，网络需要 624×192 引擎；不同序列可能需要不同尺寸。
`run.yaml` 记录实际图像和网络尺寸、校正旋转及 `T_cam0_left`。

## TUM VI

使用官方 `512_16` EuRoC/DSO 导出，传入 `--format tumvi`。
从 `dso/camchain.yaml` 读取针孔加等距畸变标定，使用 OpenCV fisheye 双目校正；
保留物理基线与相机到 IMU 的旋转和杆臂。
IMU 标定来自 `dso/imu_config.yaml`，测量优先读取 `mav0/imu0/data.csv`，
也支持 `dso/imu.txt`。16 位 PNG 使用固定右移 8 位的灰度输入，无逐帧强度归一化。
512×512 图像使用同尺寸 TensorRT 引擎。

## 时间与坐标

VIO 仅处理图像和 IMU 共同覆盖的区间，边界跳过帧记录在 `run.yaml`；
中间 IMU 缺样不外推。估计器不读取真值。
输出四元数顺序为 qx、qy、qz、qw；轨迹为相机到世界变换。
EuRoC/TUM VI 输出采用原始左相机坐标轴。
KITTI 彩色轨迹属于原始左彩色相机，不能直接等同于 KITTI cam0。
