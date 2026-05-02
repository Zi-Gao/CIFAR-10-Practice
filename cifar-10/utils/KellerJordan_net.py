## from https://github.com/KellerJordan/cifar10-airbench/

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# 1. 辅助模块 (Helper Modules)
# ----------------------------------------------------------------------------

class CustomBatchNorm(nn.BatchNorm2d):
    """
    airbench96中使用的BatchNorm版本。
    主要特点是它的仿射变换权重(gamma)是固定的(不参与训练)，
    而偏置(beta)是可训练的。
    """
    def __init__(self, num_features, eps=1e-12):
        super().__init__(num_features, eps=eps)
        # 冻结缩放参数 gamma，使其保持初始化值 1
        self.weight.requires_grad = False
        # 偏置参数 beta 默认是可训练的

class ConvGroup(nn.Module):
    """
    airbench96的核心构建块。
    这是一个包含残差连接的深度为3的卷积组。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 第一个卷积层，用于特征提取和降采样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = CustomBatchNorm(out_channels)
        
        # 残差块部分
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.norm2 = CustomBatchNorm(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.norm3 = CustomBatchNorm(out_channels)
        
        self.activation = nn.GELU()

    def forward(self, x):
        # 主路径
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        # 残差连接的输入 (identity)
        identity = x
        
        # 残差块
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.norm3(x)
        
        # 添加残差连接
        x = x + identity
        
        # 应用最终的激活函数
        x = self.activation(x)
        return x

# ----------------------------------------------------------------------------
# 2. 主模型 (Main Model)
# ----------------------------------------------------------------------------
 
class AirBench96Model(nn.Module):
    """
    airbench96模型的标准PyTorch实现。
    """
    def __init__(self, num_classes=10, scaling_factor=1/9):
        super().__init__()
        
        # 定义网络宽度
        widths = {
            'block1': 128,
            'block2': 384,
            'block3': 512,
        }
        whiten_kernel_size = 2
        # 注意: 原始代码的 whiten_width 是 2 * 3 * k**2，这里为了保持一致性
        # whiten_width = 2 * 3 * (2**2) = 24
        whiten_width = 24
        
        self.scaling_factor = scaling_factor

        # --- 网络层定义 ---

        # 1. 美白卷积层 (Whitening Layer)
        # 这是一个固定的特征提取器，权重不参与训练
        self.whitening = nn.Conv2d(3, whiten_width, kernel_size=whiten_kernel_size, padding=0, bias=True)
        self.whitening.weight.requires_grad = False # 冻结权重
        # 偏置默认是可训练的，将在训练初期学习

        self.whitening_activation = nn.GELU()

        # 2. 三个核心的ConvGroup块
        self.block1 = ConvGroup(whiten_width, widths['block1'])
        self.block2 = ConvGroup(widths['block1'], widths['block2'])
        self.block3 = ConvGroup(widths['block2'], widths['block3'])

        # 3. 分类头 (Classification Head)
        self.final_pool = nn.MaxPool2d(3) # 在3x3的特征图上池化，得到1x1
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(widths['block3'], num_classes, bias=False)
        
        # 初始化权重 (这是一个好习惯)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Dirac初始化卷积层，与原始代码保持一致
                if m.weight.requires_grad:
                    nn.init.dirac_(m.weight.data[:m.in_channels])
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Linear):
                # 使用标准的Kaiming初始化全连接层
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.whitening(x)
        x = self.whitening_activation(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.final_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        # 对最终的 logits 进行缩放
        x = x * self.scaling_factor
        
        return x

    def freeze_whitening_bias(self):
        """
        一个辅助函数，用于在训练的特定阶段冻结美白层的偏置。
        这与原始代码在3个epoch后切换模型的行为相对应。
        """
        self.whitening.bias.requires_grad = False
        print("Whitening layer bias has been frozen.")

    def initialize_whitening_layer_from_data(self, data_loader, num_images=5000, eps=5e-4):
        """
        从训练数据中计算ZCA白化矩阵并初始化美白层权重。
        这是达到高性能的关键步骤。
        """
        print("Initializing whitening layer from training data...")
        # 1. 收集图像块
        patches = []
        images_collected = 0
        for images, _ in data_loader:
            # unfold: 从图像中提取所有2x2的块
            # c=3, h=2, w=2
            p = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2)
            patches.append(p)
            images_collected += len(images)
            if images_collected >= num_images:
                break
        patches = torch.cat(patches).float()
        
        # 2. 计算协方差矩阵和特征向量
        patches_flat = patches.view(len(patches), -1)
        cov = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        # 3. 计算ZCA白化矩阵
        # 按照特征值从大到小排序
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.T.flip(0)
        
        zca_matrix = eigenvectors / torch.sqrt(eigenvalues.view(-1, 1) + eps)
        
        # 4. 设置卷积核权重
        # 卷积核的形状是 (out_channels, in_channels, kH, kW) -> (24, 3, 2, 2)
        # zca_matrix 的形状是 (12, 12)
        whitening_kernels = zca_matrix.reshape(12, 3, 2, 2)
        
        # 原始代码使用了正负两组滤波器
        self.whitening.weight.data.copy_(torch.cat([whitening_kernels, -whitening_kernels]))
        print("Whitening layer initialized successfully.")


# ----------------------------------------------------------------------------
# 3. 使用示例 (Usage Example)
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# 1. 辅助模块 (Helper Modules)
# ----------------------------------------------------------------------------

class HighMomentumBatchNorm(nn.BatchNorm2d):
    """
    airbench94_muon 中使用的BatchNorm版本。
    特点:
    - 仿射变换权重(gamma)固定，不参与训练。
    - 使用了非常高的动量 (momentum=0.6)，这意味着它对新批次的统计数据依赖更少，
      running_mean 和 running_var 更新得非常慢。
      (注意: PyTorch的momentum参数与常规定义相反，1-momentum 才是衰减因子)
    """
    def __init__(self, num_features, momentum=0.6, eps=1e-12):
        # PyTorch的momentum参数是衰减因子，所以用 1-momentum
        super().__init__(num_features, eps=eps, momentum=1 - momentum)
        # 冻结缩放参数 gamma
        self.weight.requires_grad = False
        # 偏置参数 beta 默认是可训练的

class SimpleConvGroup(nn.Module):
    """
    airbench94_muon 的核心构建块。
    这是一个不含残差连接的简单卷积组。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = HighMomentumBatchNorm(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', bias=False)
        self.norm2 = HighMomentumBatchNorm(out_channels)
        
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)
        
        return x

# ----------------------------------------------------------------------------
# 2. 主模型 (Main Model)
# ----------------------------------------------------------------------------

class AirBench94MuonModel(nn.Module):
    """
    airbench94_muon 模型的标准PyTorch实现。
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 定义网络宽度
        widths = {
            'block1': 64,
            'block2': 256,
            'block3': 256,
        }
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * (whiten_kernel_size**2) # 24

        # --- 网络层定义 ---

        # 1. 美白卷积层 (Whitening Layer)
        # 权重固定，偏置可训练
        self.whitening = nn.Conv2d(3, whiten_width, kernel_size=whiten_kernel_size, padding=0, bias=True)
        self.whitening.weight.requires_grad = False

        # 2. 核心层 (使用 nn.Sequential 保持简洁)
        self.layers = nn.Sequential(
            nn.GELU(),
            SimpleConvGroup(whiten_width, widths['block1']),
            SimpleConvGroup(widths['block1'], widths['block2']),
            SimpleConvGroup(widths['block2'], widths['block3']),
            nn.MaxPool2d(3),
        )

        # 3. 分类头 (Classification Head)
        self.head = nn.Linear(widths['block3'], num_classes, bias=False)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                # 使用Dirac初始化卷积层
                if m.weight.requires_grad:
                    # w[:w.size(1)] 确保只初始化与输入通道数相同的单位矩阵部分
                    w = m.weight.data
                    nn.init.dirac_(w[:w.size(1)])
            elif isinstance(m, nn.Linear):
                # 对分类头进行归一化
                w = m.weight.data
                w *= 1 / w.std()
            elif isinstance(m, HighMomentumBatchNorm):
                # BN层默认初始化 (gamma=1, beta=0) 已经足够好
                pass

    def forward(self, x, whiten_bias_grad=True):
        # 原始代码有一个控制美白层偏置梯度的选项
        b = self.whitening.bias if whiten_bias_grad else self.whitening.bias.detach()
        
        x = F.conv2d(x, self.whitening.weight, b)
        x = self.layers(x)
        
        # 展平特征图
        x = x.view(len(x), -1)
        
        # 应用分类头并进行缩放
        # 缩放因子是特征维度的倒数
        logits = self.head(x) / x.size(-1)
        
        return logits
    
    def freeze_whitening_bias(self):
        """辅助函数，用于冻结美白层的偏置。"""
        self.whitening.bias.requires_grad = False
        print("Whitening layer bias has been frozen.")

    def initialize_whitening_layer_from_data(self, data_loader, num_images=5000, eps=5e-4):
        """
        从训练数据中计算ZCA白化矩阵并初始化美白层权重。
        与acc96模型的实现完全相同。
        """
        print("Initializing whitening layer from training data...")
        # (这部分代码与 acc96 模型的实现相同)
        patches = []
        images_collected = 0
        for images, _ in data_loader:
            p = images.unfold(2, 2, 1).unfold(3, 2, 1).transpose(1, 3).reshape(-1, 3, 2, 2)
            patches.append(p)
            images_collected += len(images)
            if images_collected >= num_images:
                break
        patches = torch.cat(patches).float()
        
        patches_flat = patches.view(len(patches), -1)
        cov = (patches_flat.T @ patches_flat) / len(patches_flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.T.flip(0)
        
        zca_matrix = eigenvectors / torch.sqrt(eigenvalues.view(-1, 1) + eps)
        
        whitening_kernels = zca_matrix.reshape(12, 3, 2, 2)
        
        self.whitening.weight.data.copy_(torch.cat([whitening_kernels, -whitening_kernels]))
        print("Whitening layer initialized successfully.")

if __name__ == '__main__':
    # 假设我们有一个CIFAR-10的数据加载器
    # 这里用随机数据模拟
    from torch.utils.data import TensorDataset, DataLoader
    
    dummy_images = torch.randn(100, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (100,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=32)

    # 1. 实例化模型
    model = AirBench96Model(num_classes=10)
    print("Model Architecture:")
    print(model)
    
    # 2. (关键步骤) 使用训练数据初始化美白层
    model.initialize_whitening_layer_from_data(dummy_loader)

    # 3. 模拟训练过程
    print("\n--- Simulating Training ---")
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    
    # 假设这是训练的前3个epoch
    print("Training for 3 epochs (with trainable whitening bias)...")
    # ... 在这里执行训练循环 ...
    
    # 3个epoch之后，冻结美白层的偏置
    model.freeze_whitening_bias()
    
    print("\nContinuing training with frozen whitening bias...")
    # ... 在这里继续训练 ...

    # 4. 测试前向传播
    print("\n--- Testing Forward Pass ---")
    input_tensor = torch.randn(4, 3, 32, 32) # batch_size=4
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 10)
    print("Forward pass successful!")