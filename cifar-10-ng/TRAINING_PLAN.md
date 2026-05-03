# CIFAR-10 ResNet-18 训练计划 (NG版)

本项目旨在结合 **Airbench** 的极速收敛策略与 **Muon** 优化器，在 CIFAR-10 数据集上实现高精度（目标 95.5%+）的 ResNet-18 训练。

## 1. 数据处理 (Data Pipeline)
- **数据集划分：** 
  - 训练集 (Train): 45,000 张
  - 验证集 (Val): 5,000 张 (从原始训练集按 10% 比例分出)
  - 测试集 (Test): 300,000 张 (Kaggle 官方测试集)
- **存储格式：** 原始图像解压后转换为 `.npy` 数组，加速读取。
- **归一化：** 使用 CIFAR-10 均值 `(0.4914, 0.4822, 0.4465)` 和标准差 `(0.2023, 0.1994, 0.2010)`。

## 2. 模型架构 (Architecture Tweaks)
针对 32x32 小尺寸图像优化标准的 ResNet-18：
- **Conv1:** 将 `7x7, stride=2` 修改为 `3x3, stride=1, padding=1`。
- **Initial MaxPool:** 移除开头的 `MaxPool2d`。
- **Activation:** 采用 `GELU` 替代 `ReLU` 以获得更平滑的收敛。

## 3. 训练技巧 (Bag of Tricks)
### 优化器 (Optimization)
- **Muon 优化器 (PyTorch 2.11+):** 
  - 应用于 2D 权重 (卷积和全连接层)。
  - 设置: `lr=0.02`, `momentum=0.95`, `nesterov=True`。
- **AdamW 优化器:** 
  - 应用于 1D 参数 (Bias, BatchNorm 层)。
  - 设置: `lr=3e-4`, `weight_decay=0.01`。

### 数据增强 (Augmentation)
- **基础：** `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`。
- **进阶：** **Cutout (16x16)**。
- **标签平滑：** `LabelSmoothingLoss(0.1)`。

### 调度器 (Scheduler)
- **OneCycleLR:** 配合大初始学习率实现极速收敛。
- **周期：** 200 Epochs (追求极致精度) 或 40 Epochs (快速验证)。

## 4. 验证与推理 (Validation & Inference)
- **EMA:** 训练期间保持权重的指数移动平均。
- **TTA (Test-Time Augmentation):** 推理时对测试图进行水平翻转预测并求平均。

## 5. 实施步骤
1. [ ] 运行 `data_process.py` 完成数据转换。
2. [ ] 编写 `model.py` 定义优化后的 ResNet-18。
3. [ ] 编写 `train.py` 实现双优化器训练逻辑。
4. [ ] 运行训练并记录 Tensorboard 日志。
