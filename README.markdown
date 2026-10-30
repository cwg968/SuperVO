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
  --window-feedback resolve --output results/vio
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

## 启动参数完整说明

两个入口使用同一套 34 个长参数，`-h` 是 `--help` 的别名。
**只有 `--dataset` 必填**；`--help` 不要求数据集。
`superVO` 默认关闭 IMU，`superVIO` 默认开启 IMU，其他默认值相同。
参数采用 `--参数 值` 或无值开关形式，不使用 `--参数=值`。

### 必要性审查与推荐写法

- 日常运行保留输入、输出、所需推理后端与算法模式即可，不需要把所有参数写在命令中。
- TensorRT 需同时指定 `--backend tensorrt --engine FILE`；CPU 默认无需这两个参数。
  GPU 引擎已经包含模型权重，运行时不能再通过 `--weights` 更换它。
- 使用 `superVIO` 不必再写 `--imu`；KITTI 没有 IMU 输入，使用 `superVO`。
  当前没有关闭 `superVIO` 默认 IMU 的反向开关。
- 推荐滑窗仍需显式写 `--estimator window --observation-noise uniform --window-feedback resolve`。
  **程序默认值仍是 `eskf`、`patch`、`off`**，不能因省略这些参数而假定启用了推荐配置。
- `--window-states 10 --window-landmarks 300 --window-iterations 6 --window-audit-every 0`
  都等于默认值，正常命令可以省略。需要冻结完整配置或调参时再显式填写。
- 前端阈值、两类 BA 预算、审计采样与同步回环均保留为进阶控制项，普通运行无需调整。
  它们有实际用途，没有为缩短命令而删除复现或诊断能力。
- `--viewer` 和 `--realtime` 分别控制显示与播放节奏，功能独立；
  `--viewer-no-wait` 是窗口关闭策略，`--loop-sync` 是回环调度策略，均不能替代其功能开关。

### 输入、输出与数据检查

| 参数 | 默认值／取值 | 含义与生效条件 | 是否需要填写 |
|---|---|---|---|
| `--dataset PATH` | 无默认值 | 数据集序列目录；EuRoC/TUM VI 也接受 `mav0`。相对路径以当前工作目录为准。 | 必填，`--help` 除外 |
| `--format NAME` | `auto`；`auto/euroc/tumvi/kitti` | 选择输入适配器；auto 根据目录和标定文件检测格式。 | 通常省略，需明确输入格式时填写 |
| `--output DIR` | `results/run` | 轨迹、日志和运行配置目录。`--inspect` 不写输出。 | 可省略；多次运行建议分别指定新目录 |
| `--image-scale FACTOR` | `1`；`0 < FACTOR <= 1` | 缩小图像并按实际像素中心更新内参，米制基线不变；缩放后每边至少 8 像素。改变后需匹配新的 TensorRT 输入尺寸。 | 需要缩放时填写 |
| `--kitti-cameras MODE` | `auto`；`auto/gray/color` | KITTI 的左右相机选择；gray 为 image_0/1，color 为 image_2/3，auto 优先已有灰度双目。显式 gray/color 只允许 KITTI。 | KITTI 需固定相机组时填写 |
| `--inspect` | 关闭，无值开关 | 校验数据清单、标定并读取首对图像后退出；IMU 开启时也读取 IMU 标定和输入。不会加载模型、创建输出或启动估计器／窗口。 | 仅检查数据时填写 |

`--inspect` 不验证模型、引擎或可视化环境；即使提供相关参数，也不执行它们。
检查数据通常只需 `superVO --dataset PATH --inspect`；检查 IMU 输入则使用 `superVIO`。

### 推理后端与前端跟踪

| 参数 | 默认值／取值 | 含义与生效条件 | 是否需要填写 |
|---|---|---|---|
| `--backend NAME` | `cpu`；`cpu/tensorrt` | SuperPoint 推理后端；TensorRT 需构建时开启 `SUPERVO_WITH_TENSORRT`，失败时不自动退回 CPU。 | 使用 GPU 时必须写 tensorrt |
| `--weights FILE` | `weights/superpoint_v1.wts` | CPU SuperPoint 权重；仅 CPU 模式生效。路径相对工作目录，不相对可执行文件。 | CPU 使用其他权重或从其他目录启动时填写 |
| `--engine FILE` | 无默认值 | TensorRT 引擎；需匹配目标设备、SDK 和补零到 8 倍数后的网络尺寸。并非原始权重文件。 | TensorRT 模式必填，CPU 模式不接受 |
| `--max-points N` | `1000`；`N >= min-inliers` | 每幅关键帧 SuperPoint 检测上限及活动地标预算；不是滑窗后端地标上限，也不保证实际能找到 N 个点。 | 进阶调参，通常省略 |
| `--min-inliers N` | `20`；`N >= 6` | 初始化、位姿／几何验收所需的最低支持，也传给 BA 的验收门限。降低它可能接受较弱几何。 | 进阶调参，通常省略 |
| `--keyframe-interval N` | `10`；整数 `N >= 1` | 常规补点与参考刷新检查间隔；点数下降、覆盖不足可提前触发，并非严格每 N 帧建一次关键帧。保留点／模板年龄预算也随该值变化。 | 进阶调参，通常省略 |
| `--threshold SCORE` | `0.015`；`0 < SCORE < 1` | SuperPoint 检测分数门限；不是描述子距离、LK 误差或 PnP 重投影门限。 | 进阶调参，通常省略 |

### IMU 与双目惯性滑窗

下面的 `window-*` 参数及 `--observation-noise` 仅对开启 IMU 且选择 window 的运行有效。
ESKF 不使用这些像素级滑窗设置；纯视觉局部 BA 另用 `ba-*` 参数。

| 参数 | 默认值／取值 | 含义与生效条件 | 是否需要填写 |
|---|---|---|---|
| `--imu` | superVO 关闭，superVIO 开启；无值开关 | 启用惯性融合，要求标定和连续 IMU 输入；两个入口共享实现。 | superVIO 中冗余；superVO 需要惯性融合时填写 |
| `--estimator NAME` | `eskf`；`eskf/window` | IMU 估计器；ESKF 为位姿层滤波，window 为像素级紧耦合滑窗。window 需 `SUPERVO_WITH_BA=ON`。二者互斥。 | 推荐滑窗需显式写 window |
| `--observation-noise MODEL` | `patch`；`uniform/patch` | window 的像素观测噪声；uniform 为 1 px 各向同性，patch 为有界图像梯度协方差工程模型，并非学习网络。 | 推荐显式写 uniform |
| `--window-states N` | `10`；整数 `8..20` | 滑窗导航状态容量，含受保护参考状态；与前端关键帧间隔不同。 | 默认预算可省略 |
| `--window-landmarks N` | `300`；`min-inliers..2000` | 后端优化地标容量；不减少前端允许跟踪的点数。 | 默认预算可省略 |
| `--window-iterations N` | `6`；整数 `1..30` | 每次主求解的迭代上限，可提前终止；不包含 resolve 清理后的额外预算。 | 默认预算可省略 |
| `--window-feedback MODE` | `off`；`off/prune/resolve` | off 关闭后验反馈；prune 发布后验并清理持续离群观测；resolve 在实际清理后额外求解一次、最多两次迭代，再验收或回滚。 | 推荐显式写 resolve；prune 保留供对照诊断 |
| `--window-audit-every N` | `0`；整数 `N >= 0` | 只读因子／IMU 权重快照的帧采样间隔；0 关闭，正数生成额外诊断 CSV，不改变求解方程。 | 日常省略；诊断时使用如 20 |

### 纯视觉局部 BA 与可选回环

| 参数 | 默认值／取值 | 含义与生效条件 | 是否需要填写 |
|---|---|---|---|
| `--visual-ba MODE` | `none`；`none/local` | 启用纯双目局部 BA；local 需 `SUPERVO_WITH_BA=ON`，不能与 IMU 融合同时使用，应运行 superVO。 | 需要纯视觉 BA 时填写 local |
| `--ba-keyframes N` | `6`；整数 `2..12` | 纯视觉 BA 的关键帧容量；还可保留一个临时当前状态。只在 local 模式有效。 | 默认预算可省略 |
| `--ba-landmarks N` | `300`；`min-inliers..2000` | 纯视觉 BA 优化地标容量，只在 local 模式有效。 | 默认预算可省略 |
| `--ba-iterations N` | `6`；整数 `1..30` | 纯视觉 BA 求解迭代上限，只在 local 模式有效；与 window-iterations 独立。 | 默认预算可省略 |
| `--loop-closure` | 关闭，无值开关 | 启用独立的重力对齐回环位姿图，需 IMU 和 `SUPERVO_WITH_LOOP=ON`；默认异步，输出局部与全局轨迹分别保存。 | 需要回环时填写 |
| `--loop-sync` | 关闭，无值开关 | 将回环改为同步处理，便于确定性检查；可能增加前台等待，必须搭配 loop-closure。 | 进阶调度，正常可省略 |

### 显示、播放与运行控制

| 参数 | 默认值／取值 | 含义与生效条件 | 是否需要填写 |
|---|---|---|---|
| `--viewer` | 关闭，无值开关 | 打开左目特征与三维地图窗口，需 `SUPERVO_WITH_PANGOLIN=ON` 和可用显示环境；不启用回环。 | 需要实时显示时填写 |
| `--viewer-no-wait` | 关闭，无值开关 | 输入结束后自动关闭窗口，必须搭配 viewer；默认会保留窗口直到用户关闭。 | 自动运行／无需保留窗口时填写 |
| `--realtime` | 关闭，无值开关 | 按数据集时间戳等待播放；不保证处理能追上实时，不丢弃处理不过来的输入。与 viewer 独立，暂停恢复不会快进补偿。 | 希望按采集节奏播放时填写 |
| `--max-frames N` | `0`；整数 `N >= 0` | 限制同步双目输入帧数，0 为全部。限制发生在 IMU 首尾覆盖裁剪前，实际输出可能少于 N。 | 短段检查时填写，完整运行省略 |
| `--threads N` | `4`；整数 `N >= 1` | OpenCV 的 CPU 线程数；不是总进程线程上限，也不设置 TensorRT batch 或 Ceres 求解线程。 | 需要固定 CPU 资源时填写 |
| `--overwrite` | 关闭，无值开关 | 允许覆盖已有的已知输出文件；不会清空目录或删除无关文件，建议新运行使用新目录。 | 确实要复用输出目录时填写 |
| `--help` / `-h` | 无值开关 | 打印当前入口的参数分组、默认值和依赖后退出，不要求 dataset。 | 查看帮助时填写 |

### 启动组合校验

正常运行现在会拒绝显式提供但不生效的组合，而不是悄悄忽略：

- `--engine` 搭配 CPU，或 TensorRT 缺少 engine；TensorRT 模式显式传 `--weights`。
- 未开启 IMU 却指定 estimator；ESKF／纯视觉模式显式传 window 参数或 observation-noise。
- 未开启 `--visual-ba local` 却传 ba 参数；纯视觉 BA 与 IMU 同时启用。
- 缺少 loop-closure 的 loop-sync、缺少 viewer 的 viewer-no-wait。
- 非 KITTI 输入选择 kitti-cameras gray/color，或参数超出表中的合法范围。

这些检查不改变算法默认值。重复传同一有值参数时，最后一个值生效；
不认识的参数与缺失值会直接报错。`--inspect` 是数据检查模式，不执行上述推理／显示模块。

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
