import torch
from torchvision import transforms

normalize=transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.4914009, 0.48215896, 0.4465308 ], std=[0.24703279, 0.24348423, 0.26158753]),
])