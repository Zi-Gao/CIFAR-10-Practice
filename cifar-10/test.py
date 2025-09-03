import os
import math
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ReduceLROnPlateau

from muon import SingleDeviceMuon

import numpy as np
from tqdm import tqdm,trange
from torch.utils.tensorboard import SummaryWriter
import torchvision

from utils import read_data
from utils.resnet import resnet18
from utils.normalize import normalize


print("\n1. Configuring device...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU instead.")

torch.set_default_dtype(torch.float32)


NUM_CLASS=10
test_batch_size=512

test_pic=read_data.get_test_pic()
test_images=torch.from_numpy(test_pic)

test_dataset=TensorDataset(test_images)
test_loader=DataLoader(test_dataset,batch_size=test_batch_size,shuffle=False)
print(type(test_dataset[0]))

label_name=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']



model = resnet18(num_classes=NUM_CLASS).to(device)
checkpoint = torch.load('./run/resnet18/7/ckpt/best.pth.tar', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

i=1

f=open("result.csv","w")
f.write("id,label\n")

for image, in tqdm(test_loader):
    def run(image):
        image=normalize(image).to(device)
        output=model(image)
        _, predicted_indices = torch.max(output, 1) # 返回最大值和对应的索引
        return predicted_indices
    
    predicted_indices=run(image)

    for pred_index in predicted_indices:
        label = label_name[pred_index.item()]
        f.write(f"{i},{label}\n")
        i += 1


## kaggle competitions submit -c cifar-10 -f result.csv -m "ResNet18+Muon+300epoch"