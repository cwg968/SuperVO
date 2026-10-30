# SuperVO / SuperVIO

基于 SuperPoint 的 C++17 双目视觉与视觉惯性里程计，支持 EuRoC、KITTI 和 TUM VI。
提供 CPU / TensorRT 10.x 推理、Pangolin 实时可视化、无界面运行和轨迹输出。

## 功能

- 双目 SuperPoint 描述子匹配、极线亚像素细化与左右目独立 LK 跟踪。
- 双目 PnP 选择、联合重投影恢复、曝光补偿和有界关键帧模板重关联。
- 可选纯双目局部 BA，或独立的 ESKF / 紧耦合双目惯性滑窗后端。
- 滑窗联合优化位姿、速度、偏置、地标与固定模长重力方向，使用 QR 平方根边缘化。
- 后验持续离群清理与有界条件重求解；可选只读残差、IMU 权重诊断。
- 可选独立回环与 4DoF 位姿图；默认关闭。
- 实时左目图像与特征点叠加、相机视锥、关键帧、轨迹及稀疏三维地图。

## 构建

依赖 CMake >= 3.18、C++17、Eigen >= 3.3 与 OpenCV 4
（core、imgproc、imgcodecs、calib3d、video、dnn、features2d），
以及 [Pangolin](https://github.com/stevenlovegrove/Pangolin)（OpenGL 可视化，默认构建）。
滑窗或回环另需 Ceres >= 2.1；GPU 推理另需 CUDA 与 TensorRT 10.x。

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
build/superVO --help
```

启用滑窗和 TensorRT：

```sh
cmake -S . -B build-ba -DCMAKE_BUILD_TYPE=Release \
  -DSUPERVO_WITH_BA=ON -DSUPERVO_WITH_TENSORRT=ON \
  -DTENSORRT_ROOT=/path/to/TensorRT
cmake --build build-ba -j2
build-ba/supervo_build_engine weights/superpoint_v1.wts build-ba/euroc.engine 752 480
```

私有安装的依赖可用 `CMAKE_PREFIX_PATH`、`Ceres_DIR`、`OpenCV_DIR`、`Pangolin_DIR` 指定。
Pangolin 安装到标准前缀时自动查找；纯无界面构建可传 `-DSUPERVO_WITH_PANGOLIN=OFF`。
其他构建选项及 GPU 安装说明见 [构建与 TensorRT](docs/BUILD.md)。

## 运行

CPU 双目视觉：

```sh
build/superVO --dataset /path/to/EuRoC/sequence --output results/vo
build/superVO --dataset /path/to/KITTI/sequences/00 --format kitti --output results/kitti00
```

推荐的 GPU 双目惯性滑窗配置：

```sh
build-ba/superVIO --dataset /path/to/EuRoC/sequence \
  --backend tensorrt --engine build-ba/euroc.engine \
  --estimator window --observation-noise uniform \
  --window-states 10 --window-landmarks 300 --window-iterations 6 \
  --window-feedback resolve --window-audit-every 0 --output results/vio
```

`superVIO` 默认启用 IMU，但后端默认值仍为 `eskf`；使用滑窗时需显式传入
`--estimator window`。`resolve` 在实际清理观测后额外求解最多一次、两次迭代。
不传 `--loop-closure` 即关闭回环。完整参数见 `--help`。

TUM VI 使用 `--format tumvi` 与 512×512 引擎。KITTI 无 IMU；
彩色输入和缩放示例见 [数据输入](docs/DATASETS.md)。

## 实时可视化

在上面的运行命令末尾加 `--viewer` 即可打开 Pangolin 窗口；
建议同时加 `--realtime`，按数据集时间戳播放。

```sh
build-ba/superVIO --dataset /path/to/EuRoC/sequence \
  --backend tensorrt --engine build-ba/euroc.engine \
  --estimator window --observation-noise uniform \
  --window-states 10 --window-landmarks 300 --window-iterations 6 \
  --window-feedback resolve --viewer --realtime --output results/vio-viewer
```

- 左侧为估计器实际处理的校正左目图像：绿色为当前实测跟踪点，橙黄色为当前关键帧新提取的 SuperPoint 点。
  普通帧复用 LK 跟踪，不为显示额外运行特征网络，也不把投影位置伪装成实测点。
- 右侧显示局部里程计坐标系中的相机、关键帧视锥、橙色轨迹和稀疏地图。
  绿色为当前地图点，灰蓝色为累计的最近一次估计；历史点不是经全局 BA 修正的稠密地图。
  即使开启回环，此窗口仍展示局部里程计，避免混用全局校正与局部点坐标。
- 鼠标拖动、滚轮可查看三维场景；面板提供跟随相机、显示开关、暂停和视角复位。
  取消跟随后可以自由查看地图。
- 输入结束后保留窗口，关闭窗口退出；运行中关闭窗口会停止继续读取输入并保存已有轨迹。
  `--viewer --viewer-no-wait` 在输入结束时自动关闭。

渲染线程只接收只读快照，显示帧允许丢弃；缓存上限为 50,000 个地图点、50,000 个轨迹采样和
2,000 个关键帧。`--viewer` 不改变估计方程，但复制和渲染会占用资源；耗时面板不含显示开销。
默认不打开窗口，服务器或无显示环境可直接运行原命令。

## 输出

- `trajectory.tum`：时间戳、位置与四元数，相机到世界变换。
- `frames.csv`：视觉状态、跟踪、恢复和处理耗时。
- `inertial.csv`：初始化、传播、偏置、重力和零速状态。
- `backend.csv`：滑窗支持、求解、离群清理和边缘化诊断。
- `run.yaml`：实际参数、相机标定与推理后端。

已有输出目录默认受保护，只有 `--overwrite` 才允许覆盖。
`--max-frames N` 限制输入帧数，`--inspect` 只检查输入。
轨迹可能因输入或跟踪失败提前停止，应同时检查退出码和输出状态。

## 开发格式

C++ 源码采用仓库的 `.clang-format`，使用 clang-format 18：

```sh
git ls-files -z '*.cpp' '*.h' | xargs -0 clang-format -i
```

## 来源与许可

SuperPoint 模型与预训练权重来源：[Magic Leap SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)。
预积分原理参考 [Forster 等](https://arxiv.org/abs/1512.02363)，
平方根边缘化参考 [Demmel 等](https://arxiv.org/abs/2109.02182)。
项目代码许可见 [LICENSE](LICENSE)，第三方依赖与模型遵循各自许可。

运行架构、离群清理和诊断边界见 [算法与运行说明](docs/PIPELINE.md)。
