# CIFAR-10 极速高精度训练 (ResNet-18 & AirBench)

本项目针对 **CIFAR-10** 数据集，结合 **Airbench** 的极速收敛架构设计与前沿的 **Muon** 优化器策略，实现了在较短训练周期内的高精度收敛。

当前模型在验证集/测试集上已达到 **95.64% (0.95640 Acc)** 的优异成绩。

---

## 核心训练特点

### 1. 混合双优化器体系 (Muon + AdamW)
- **高维权重采用 Muon 优化器**：
  - 针对 Conv2d 卷积核（4D 张量）应用基于 Newton-Schulz 5次多项式迭代的正交化更新器（`zeropower_via_newtonschulz5`）。
  - 配合参数范数归一化（Weight Normalization），保持正交更新步长稳定，显著加快特征学习效率。
- **1D 参数采用 AdamW / SGD**：
  - 对 Bias、BatchNorm 参数以及线性层等 1D 参数使用标准的 AdamW / SGD 优化器，分工协作。

### 2. 图像适配与网络架构设计
- **输入特征白化 (Patch Whitening)**：
  - 针对 32×32 图像特性，首层采用基于训练集局部 patch 协方差特征分解初始化的白化卷积层，极速提取图像的无相关正交特征；
  - 前 5 个 epoch 后冻结白化偏置梯度，稳定基础特征表示。
- **高效 Backbone 与平滑激活**：
  - 针对 32×32 小尺寸裁剪优化（移除冗余降采样与大卷积核）；
  - 全面使用 **GELU** 激活函数代替传统 ReLU，获得更平滑的损失曲面与梯度流。

### 3. 高效数据流与强正则化
- **内存级加速 Pipeline**：
  - 原始图像预处理离线存为 `.npy` 数组并预先做标准化（Mean & Std），极大降低训练期间的 I/O 与解码开销。
- **数据增强组合**：
  - `RandomCrop(32, padding=4)` + `RandomHorizontalFlip`；
  - **Cutout (16×16 单孔擦除)** 遮挡增强，强迫网络学习上下文互补特征。
- **标签平滑 (Label Smoothing)**：
  - 采用 $\epsilon = 0.1$ 的 Label Smoothing 交叉熵损失，抑制过拟合并防止过度自信。

### 4. 训练调度与推理提升 (Bag of Tricks)
- **OneCycleLR 调度**：配合较高初始学习率实现极速热身与余弦退火收敛。
- **EMA (Exponential Moving Average)**：衰减率 0.999 的指数移动平均，过滤权重随机震荡，获得更强泛化力。
- **TTA (Test-Time Augmentation)**：测试与评估阶段融合镜像翻转（Mirror）与轻微反射平移（Translation），显著提升预测稳定性与准确率。

---

## 性能表现

| 阶段 | 周期 (Epochs) | 验证/测试准确率 (Acc) | 特性 |
| :--- | :---: | :---: | :--- |
| **已达成果** | **~45** | **95.640% (0.95640)** | Muon + AdamW + Whitening + EMA + TTA |

---

## 目录结构

```text
├── data/                  # CIFAR-10 原始与处理后的 .npy 数据
├── utils/
│   ├── muon.py            # Muon 优化器与 Newton-Schulz 正交化实现
│   ├── data_utils.py      # Npy 数据集加载器与 Cutout 增强
│   └── inference.py       # EMA 权重跟踪与 TTA 评估函数
├── model.py               # 优化版 ResNet-18 与 AirBench96 架构定义
├── train.py               # 双优化器训练与验证主入口
├── test.py                # 测试集推理与 Kaggle submission 导出
├── data_process.py        # 图像数据转换为 .npy 格式
└── README.md
```

---

## 快速使用

### 1. 数据准备
将 CIFAR-10 解压到 `data/` 目录后运行：
```bash
python data_process.py
```

### 2. 模型训练
```bash
python train.py --epochs 45 --opt1d adamw --use_ema
```
训练过程支持通过 TensorBoard 查看 Loss 与 Accuracy 实时曲线：
```bash
tensorboard --logdir runs
```

### 3. 测试与预测
使用训练好的权重进行 TTA 测试并生成提交文件：
```bash
python test.py --ckpt runs/<run_name>/best_model.pth
```
