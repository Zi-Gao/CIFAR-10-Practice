import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Standard ResNet-18 (No manual reshaping needed now) ---

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.gelu = nn.GELU()
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
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


# --- AirBench96 Architecture (Optimized for Muon & Speed) ---

class AirBenchBatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12, weight=False, bias=True):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class AirBenchConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class AirBenchConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth):
        super().__init__()
        assert depth in (2, 3)
        self.depth = depth
        self.conv1 = AirBenchConv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = AirBenchBatchNorm(channels_out)
        self.conv2 = AirBenchConv(channels_out, channels_out)
        self.norm2 = AirBenchBatchNorm(channels_out)
        if depth == 3:
            self.conv3 = AirBenchConv(channels_out, channels_out)
            self.norm3 = AirBenchBatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.depth == 3:
            x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        if self.depth == 3:
            x = self.conv3(x)
            x = self.norm3(x)
            x = x + x0
            x = self.activ(x)
        return x

class AirBench96(nn.Module):
    def __init__(self, num_classes=10, widths=None, depth=3, scaling_factor=1/9):
        super().__init__()
        if widths is None:
            widths = {'block1': 128, 'block2': 384, 'block3': 512}
        
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        
        self.whiten = AirBenchConv(3, whiten_width, whiten_kernel_size, padding=0, bias=True)
        self.whiten.weight.requires_grad = False # Whiten weights are fixed after initialization
        
        self.layers = nn.Sequential(
            nn.GELU(),
            AirBenchConvGroup(whiten_width,     widths['block1'], depth),
            AirBenchConvGroup(widths['block1'], widths['block2'], depth),
            AirBenchConvGroup(widths['block2'], widths['block3'], depth),
            nn.MaxPool2d(3),
        )
        
        self.head = nn.Linear(widths['block3'], num_classes, bias=False)
        self.scaling_factor = scaling_factor
        
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        # Ensure head weight initialization scale matches airbench style
        if hasattr(self.head, 'weight'):
            w = self.head.weight.data
            w *= 1 / w.std()

    def forward(self, x, whiten_bias_grad=True):
        # Whiten bias can have its gradient detached after several epochs
        b = self.whiten.bias
        x = F.conv2d(x, self.whiten.weight, b if whiten_bias_grad else b.detach())
        
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.head(x) * self.scaling_factor
        return x

# --- Whitening Initialization Utils (For AirBench only) ---

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    # Move to CPU for eigh as it's not implemented on MPS
    device = est_patch_covariance.device
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance.cpu(), UPLO='U')
    eigenvalues = eigenvalues.to(device)
    eigenvectors = eigenvectors.to(device)
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    """
    Initializes the first layer of AirBench with whitening parameters.
    train_set: a tensor of images (N, C, H, W)
    """
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

if __name__ == "__main__":
    model = resnet18()
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"ResNet18 Output shape: {y.shape}")
    
    model_ab = AirBench96()
    y_ab = model_ab(x)
    print(f"AirBench96 Output shape: {y_ab.shape}")
