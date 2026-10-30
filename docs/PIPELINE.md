# 算法与运行说明

## 模块边界

依赖方向为 runtime → system → frontend / backend / inertial。
视觉前端不依赖 Ceres；滑窗不读取图像、文件、数据集或真值。
`VisualMeasurements` 传递帧号、时间戳、地标 ID、左右目像素和协方差。
双目深度仅作三维初值，不作为独立测量重复融合。

前端使用 SuperPoint 双向描述子匹配、极线细化、左右目时间 LK 和 PnP。
曝光或局部光度恢复仍需通过几何验收。两目独立候选失败后，可联合优化双目重投影，
再尝试有界近期模板重关联。候选状态只在验收后提交。

## 惯性后端

`--estimator eskf` 为位姿层误差状态滤波；`--estimator window` 为独立的像素级滑窗。
两个后端互斥，不重复融合。
滑窗优化旋转、位置、速度、双偏置、地标与固定模长重力方向。
使用中点 IMU 预积分、完整 15×15 协方差和偏置修正，必要时重积分。
逐轴采样插值误差为工程建模项，不代表完成真实传感器噪声标定。

Ceres 求解双目重投影、惯性和先验因子，视觉使用 Huber 损失。
参考帧受保护，窗口有界；QR 消元形成平方根先验，不将阻尼作为历史信息保存。
仍存活地标在已删除非宿主帧上的观测被舍弃，以维持当前先验与 Schur 结构；
这是有信息损失的有界近似。

推荐显式使用 `uniform`、10 状态、300 地标、6 次主迭代和 `resolve`。
`patch` 为有界图像梯度协方差模型，不是学习型不确定性预测。

## 后验离群反馈

`--window-feedback off|prune|resolve` 默认 `off`。
按帧、相机、地标记录后验重投影误差。
连续两次已接受更新仍离群才允许清理；每次最多 32 个观测且不超过 10%，
每点至少保留两个约束。`resolve` 仅在实际删除后额外求解一次，最多两次迭代。
最终几何验收失败会回滚整个视觉更新，离群历史只随最终接受状态推进一次。
主求解与清理求解分别记录成本，因为二者因子集合不同。

## 恢复与静止

短暂惯性传播受实际时间和相对参考协方差共同约束，保留交叉项；
超过恢复预算停止，不能将惯性续航等同于视觉恢复。
图像与 IMU 联合静止检测通过后，滑窗可添加经创新检验的零速因子。
不将同一批 IMU 均值再作为独立偏置或重力测量。

## 只读诊断

`--window-audit-every N` 每 N 帧采样，0 为关闭。
输出 `window_residuals.csv` 和 `window_imu_weights.csv`，记录视觉、惯性白化行组、
先验、零速约束的维数与能量，以及实际噪声密度、协方差和完整白化误差。
诊断不修改估计状态。
完整白化会混合物理坐标，白化行组不是独立物理误差；
相邻快照重复包含因子，不能直接将池化能量当作独立卡方检验。

## 原理参考

- [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
- [On-Manifold Preintegration](https://arxiv.org/abs/1512.02363)
- [Square Root Marginalization](https://arxiv.org/abs/2109.02182)
- [Ceres Solver](https://ceres-solver.org/)
- [OpenVINS 零速更新](https://docs.openvins.com/update-zerovelocity.html)

这些对应模型、优化原理和设计参考，不表示移植了外部系统的完整实现。
