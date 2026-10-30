# 构建与 TensorRT

## 构建选项

| 选项 | 默认 | 用途 |
|---|---|---|
| `SUPERVO_WITH_TENSORRT` | OFF | TensorRT 10.x 推理与引擎构建工具 |
| `SUPERVO_WITH_BA` | OFF | 双目局部 BA 和双目惯性滑窗 |
| `SUPERVO_WITH_LOOP` | OFF | 回环检索、几何验证和独立位姿图 |

标准 CPU 构建只需 C++17、CMake、Eigen 和 OpenCV。
启用 BA 或回环时 CMake 查找 Ceres >= 2.1。

## GPU 推理

需要匹配的 CUDA 与 TensorRT 10.x SDK。使用 `-DTENSORRT_ROOT=/path/to/TensorRT`
指定 SDK，必要时用 `CUDAToolkit_ROOT` 指定 CUDA。

仓库提供可选的 `python3 tools/install_tensorrt.py`：从配置的 APT 源下载锁定版本的包，
校验 SHA256，再解压至用户目录，不修改驱动或系统软件包。
该工具针对 Linux 与锁定的 TensorRT 10.13.3.9 / CUDA 13.0 组合；
其他平台应安装适合本机的 NVIDIA SDK。
包版本与散列见 [锁定清单](../tools/tensorrt-packages.lock.json)。

启用 TensorRT 后生成 `supervo_build_engine`：

```sh
build-ba/supervo_build_engine weights/superpoint_v1.wts build-ba/euroc.engine 752 480
build-ba/supervo_build_engine weights/superpoint_v1.wts build-ba/tumvi.engine 512 512
```

引擎采用 batch 1、FP32；需在目标设备上构建，且尺寸应匹配网络输入。
图像右侧和底部会补零至 8 的倍数，几何仍使用有效图像区域。
例如 620×188 输入需要 624×192 引擎。
引擎加载时检查名称、类型、形状与张量模式。不同 SDK 或设备的旧引擎不能假定兼容。
同一个检测器实例不支持并发推理调用。

## 可选回环

`-DSUPERVO_WITH_LOOP=ON` 构建回环模块和 `supervo_inspect_loops`。
运行时通过 `--loop-closure` 显式开启，`--loop-sync` 选择同步执行。
回环使用独立位姿图，不将全局校正反馈至局部滑窗和边缘化先验。

纯视觉局部 BA 通过 `--visual-ba local` 开启，不能与 IMU 估计器同时使用。
