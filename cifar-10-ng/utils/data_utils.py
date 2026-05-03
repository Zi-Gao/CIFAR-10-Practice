import torch
import torchvision.transforms as T
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

# Parameters from TRAINING_PLAN.md
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        # img is a tensor (C, H, W)
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def get_train_transform():
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        Cutout(n_holes=1, length=16)
    ])

def get_val_transform():
    return T.Compose([
        T.ToTensor(),
        T.Normalize(MEAN, STD)
    ])

class CIFAR10NpyDataset(Dataset):
    def __init__(self, images_path, labels_path=None, transform=None):
        self.images = np.load(images_path)
        self.labels = np.load(labels_path) if labels_path else None
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image in (C, H, W) uint8
        img = self.images[idx]
        # Convert to PIL for torchvision transforms (H, W, C)
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        
        if self.labels is not None:
            return img, torch.tensor(self.labels[idx], dtype=torch.long)
        return img
