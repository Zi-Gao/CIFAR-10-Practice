import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset

# Parameters from TRAINING_PLAN.md
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img is a tensor (C, H, W)
        c, h, w = img.shape
        for n in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            img[:, y1:y2, x1:x2] = 0.
        return img

def get_train_transform():
    """Tensor-only fast transforms."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        Cutout(n_holes=1, length=16)
    ])

class IdentityTransform:
    def __call__(self, x):
        return x

def get_val_transform():
    return IdentityTransform()

class CIFAR10NpyDataset(Dataset):
    def __init__(self, images_path, labels_path=None, transform=None, device='cpu'):
        """
        Loads entire dataset into memory as a Tensor. 
        In Unified Memory (MPS), this is extremely efficient.
        """
        # Load and convert to (N, C, H, W) float32 [0, 1]
        raw_data = np.load(images_path)
        
        # Ensure (N, C, H, W)
        if raw_data.shape[1] != 3:
            raw_data = raw_data.transpose(0, 3, 1, 2)
            
        self.images = torch.from_numpy(raw_data).to(torch.float32) / 255.0
        
        # Pre-normalize the entire dataset once
        mean = torch.tensor(MEAN).view(1, 3, 1, 1)
        std = torch.tensor(STD).view(1, 3, 1, 1)
        self.images = (self.images - mean) / std
        
        # Move to target device (e.g., 'mps')
        self.images = self.images.to(device)
        
        if labels_path:
            self.labels = torch.from_numpy(np.load(labels_path)).to(torch.long).to(device)
        else:
            self.labels = None
            
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Since it's in unified memory, this is just a pointer/view operation
        img = self.images[idx]
        
        if self.transform:
            # We clone to avoid modifying the in-memory master dataset during augmentation
            img = self.transform(img.clone())
            
        if self.labels is not None:
            return img, self.labels[idx]
        return img
