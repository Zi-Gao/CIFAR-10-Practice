import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision import models
from typing import Type, List, Union

# 1. 自定义 ResNet 类 (与之前版本完全相同，无需修改)
class ResNetWithResize(ResNet):
    """
    一个继承自 torchvision ResNet 的模型，
    它在 forward pass 的开始处自动将输入图片 resize 到 224x224。
    """
    def __init__(self, *args, **kwargs):
        super(ResNetWithResize, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        重写 forward 方法，先 resize，再调用父类的前向传播。
        """
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        return super(ResNetWithResize, self).forward(x)

# 2. 更新后的工厂函数，增加了对 num_classes 的处理
def _create_resnet_with_resize(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    weights_enum,
    num_classes: int,
    **kwargs
) -> ResNetWithResize:
    """
    一个通用的创建函数，用于构建带有 resize 功能的 ResNet。
    """
    # 如果不使用预训练权重，直接用指定的 num_classes 创建模型
    if not pretrained:
        return ResNetWithResize(block, layers, num_classes=num_classes, **kwargs)

    # --- 处理预训练权重的情况 ---
    
    # a. 先创建一个带有原始类别数（1000）的模型，以便加载权重
    #    注意：这里我们用 num_classes=1000 来确保 fc 层的尺寸匹配
    model = ResNetWithResize(block, layers, num_classes=1000, **kwargs)
    
    # b. 加载官方的预训练权重
    state_dict = models.get_weight(weights_enum).get_state_dict(progress=True)
    model.load_state_dict(state_dict)

    # c. 如果用户指定的类别数不是 1000，则替换掉最后一层
    if num_classes != 1000:
        # 获取全连接层的输入特征数
        num_ftrs = model.fc.in_features
        # 创建一个新的全连接层并替换掉原来的
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(f"预训练模型加载完成。已将最后的分类层替换为输出 {num_classes} 个类别的新层。")

    return model


def resnet18_with_resize(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> ResNetWithResize:
    """
    构建一个带有自动 resize 功能的 ResNet-18 模型。
    
    :param pretrained: 如果为 True，返回在 ImageNet 上预训练的模型权重。
    :param num_classes: 输出的类别数。如果 pretrained=True，这会替换掉预训练模型的头。
    """
    return _create_resnet_with_resize(
        BasicBlock, [2, 2, 2, 2], pretrained,
        "ResNet18_Weights.IMAGENET1K_V1", num_classes, **kwargs
    )

def resnet50_with_resize(pretrained: bool = False, num_classes: int = 1000, **kwargs) -> ResNetWithResize:
    """
    构建一个带有自动 resize 功能的 ResNet-50 模型。
    
    :param pretrained: 如果为 True，返回在 ImageNet 上预训练的模型权重。
    :param num_classes: 输出的类别数。如果 pretrained=True，这会替换掉预训练模型的头。
    """
    return _create_resnet_with_resize(
        Bottleneck, [3, 4, 6, 3], pretrained,
        "ResNet50_Weights.IMAGENET1K_V2", num_classes, **kwargs
    )


# 3. 使用示例
if __name__ == '__main__':
    # 示例 1: 创建一个预训练的 ResNet-18，并将其用于一个 10 分类任务
    # 这会自动加载 ImageNet 权重，并替换最后的分类层
    print("--- 示例 1: 迁移学习，10个类别 ---")
    model_10_classes = resnet18_with_resize(pretrained=True, num_classes=10)
    model_10_classes.eval()
    
    # 检查最后一层是否正确
    print(f"模型最后一层: {model_10_classes.fc}")

    # 输入一个非 224x224 的图片张量
    dummy_input_1 = torch.randn(4, 3, 128, 192) # Batch size=4, H=128, W=192
    print(f"\n输入张量形状: {dummy_input_1.shape}")

    with torch.no_grad():
        output_1 = model_10_classes(dummy_input_1)
    
    print(f"模型成功处理了输入！")
    print(f"输出张量形状: {output_1.shape}") # 应该为 (4, 10)
    assert output_1.shape == (4, 10)
    
    print("\n" + "="*40 + "\n")

    # 示例 2: 创建一个没有预训练权重的 ResNet-50，用于 200 个类别的任务
    print("--- 示例 2: 从零开始训练，200个类别 ---")
    model_200_classes = resnet50_with_resize(pretrained=False, num_classes=200)
    model_200_classes.eval()
    
    print(f"模型最后一层: {model_200_classes.fc}")

    dummy_input_2 = torch.randn(2, 3, 500, 500)
    print(f"\n输入张量形状: {dummy_input_2.shape}")

    with torch.no_grad():
        output_2 = model_200_classes(dummy_input_2)

    print(f"模型成功处理了输入！")
    print(f"输出张量形状: {output_2.shape}") # 应该为 (2, 200)
    assert output_2.shape == (2, 200)