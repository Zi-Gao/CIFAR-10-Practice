# by gemini 2.5 pro

import torch
import torch.nn as nn
from typing import Type, List, Optional

# 1. 实现基础残差块 (BasicBlock)
class BasicBlock(nn.Module):
    """
    ResNet-18/34 使用的基础残差块。
    包含两个 3x3 的卷积层。
    """
    expansion = 1  # 输出通道数相对于输入通道数的扩展倍数

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 快捷连接（Shortcut/Residual Connection）
        self.downsample = None
        # 当输入输出维度不匹配时（通道数或尺寸），需要对输入 x 进行变换以匹配主路径输出的维度
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 主路径前向传播
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果有下采样层，则对输入 x 进行变换
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 将主路径的输出与快捷连接的输出相加
        out += identity
        out = self.relu(out)
        
        return out

# 2. 实现完整的 ResNet 类
class ResNet(nn.Module):
    def __init__(
        self, 
        block: Type[BasicBlock], 
        layers: List[int], 
        num_classes: int = 1000, 
        zero_init_residual: bool = False
    ):
        super(ResNet, self).__init__()
        
        self.in_channels = 64 # 第一个残差块的输入通道数
        
        # 输入部分 (Stem)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个残差阶段 (Layers)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=4)
        
        # 输出部分 (Head)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出尺寸为 (1, 1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # 将残差分支的最后一个 BN 层的 gamma 初始化为0，可以使得模型在初始阶段更容易训练
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[BasicBlock], out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        构建一个残差阶段（由多个残差块组成）。
        """
        layers = []
        # 第一个残差块可能需要进行下采样
        layers.append(block(self.in_channels, out_channels, stride))
        
        self.in_channels = out_channels * block.expansion
        
        # 后续的残差块不进行下采样
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) # 展平
        x = self.fc(x)
        
        return x

# 3. 工厂函数，创建 ResNet-18 模型
def resnet18(num_classes: int = 1000) -> ResNet:
    """
    返回一个 ResNet-18 模型实例。
    ResNet-18 的结构是 [2, 2, 2, 2] 个 BasicBlock。
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)