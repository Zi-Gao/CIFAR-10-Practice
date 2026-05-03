import torch
import torch.nn as nn
import torch.nn.functional as F

class MuonLayer:
    """Mixin for layers that need to support Muon by having 2D weights."""
    def _reshape_weight_to_2d(self):
        if self.weight.ndim > 2:
            self.register_buffer('weight_shape', torch.tensor(self.weight.shape))
            self.weight = nn.Parameter(self.weight.data.view(self.weight.size(0), -1))

class MuonConv2d(nn.Conv2d, MuonLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reshape_weight_to_2d()

    def forward(self, x):
        weight = self.weight.view(tuple(self.weight_shape.tolist()))
        return self._conv_forward(x, weight, self.bias)

class MuonLinear(nn.Linear, MuonLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Linear weights are already 2D, but we use the mixin for consistency
        pass

def conv3x3(in_planes, out_planes, stride=1):
    return MuonConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return MuonConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.gelu = nn.GELU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.gelu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = MuonConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = MuonLinear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Note: kaiming_normal_ expects the weight as it is
                # But our weight is now 2D. 
                # For Conv2d, it was (out, in*k*k), which is equivalent to Linear (out, in_features).
                # kaiming_normal_ handles 2D just fine.
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

if __name__ == "__main__":
    model = resnet18()
    # Verify all weights are 2D
    all_2d = True
    for name, p in model.named_parameters():
        if 'weight' in name and p.ndim != 2:
            print(f"Non-2D weight: {name} {p.shape}")
            all_2d = False
    print(f"All weights are 2D: {all_2d}")
    
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
