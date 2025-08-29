import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# 从你的工具文件中导入数据读取函数和模型
from utils import read_data
from utils.resnet import resnet18

# 设置默认的浮点数类型为 float32
torch.set_default_dtype(torch.float32)

# ==============================================================================
# 0. 加载数据
# ==============================================================================
print("0. Loading data...")
NUM_CLASS = 10
# 假设 read_data 返回的是 (N, C, H, W) 格式的 uint8 NumPy 数组
train_pic = read_data.get_train_pic_train()
train_label = read_data.get_train_label_train()
test_pic = read_data.get_train_pic_test()
test_label = read_data.get_train_label_test()

print(f"Train pictures shape: {train_pic.shape}, dtype: {train_pic.dtype}")
print(f"Train labels shape: {train_label.shape}")
print(f"Test pictures shape: {test_pic.shape}, dtype: {test_pic.dtype}")
print(f"Test labels shape: {test_label.shape}")

# ==============================================================================
# 1. 设备配置 (MPS)
# ==============================================================================
print("\n1. Configuring device...")
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device.")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU instead.")

# ==============================================================================
# 2. 数据准备 (不进行预归一化)
# ==============================================================================
print("\n2. Preparing data...")
# 将 NumPy 数组转换为 PyTorch Tensor，保持 uint8 类型
train_images = torch.from_numpy(train_pic)
train_labels = torch.from_numpy(train_label).long()
test_images = torch.from_numpy(test_pic)
test_labels = torch.from_numpy(test_label).long()

batch_size = 64
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("Data preparation complete. Tensors are kept as uint8 for augmentation.")

# ==============================================================================
# 2.5. 数据增强和转换设置
# ==============================================================================
print("\n2.5. Setting up data augmentation and conversion pipeline...")
# 训练集的增强管道
# 1. RandAugment (在 uint8 上操作)
# 2. Cutout (在 uint8 上操作)
# 3. 转换为 float32 并自动归一化到 [0, 1]
train_transform = transforms.Compose([
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
    transforms.ConvertImageDtype(torch.float32) # 关键步骤：转换类型并归一化
])

# 测试集的转换管道 (没有增强，只有转换和归一化)
test_transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32)
])

print("Pipelines created.")

# ==============================================================================
# 3. 模型、损失函数、优化器设置
# ==============================================================================
print("\n3. Setting up model and training components...")
model = resnet18(num_classes=NUM_CLASS).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Setup complete.")

# ==============================================================================
# 4. 训练循环 (全精度，无 autocast)
# ==============================================================================
print("\n4. Starting training in full FP32 precision...")
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, labels in progress_bar:
        # 1. 在 CPU 上应用增强和转换 (输入是 uint8, 输出是 float32)
        images = train_transform(images)
        
        # 2. 将处理好的数据移动到 MPS 设备
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播 (全精度)
        outputs = model(images)
        loss = criterion(outputs, labels)
            
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if torch.isnan(loss):
            print("\nLoss is NaN. Stopping training.")
            exit()
        
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

print("Training finished.")

# ==============================================================================
# 5. 评估模型
# ==============================================================================
print("\n5. Evaluating model on test data...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        # 1. 在 CPU 上转换类型和归一化
        images = test_transform(images)
        
        # 2. 移动到设备
        images = images.to(device)
        labels = labels.to(device)
        
        # 全精度前向传播
        outputs = model(images)
            
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on the test set: {accuracy:.2f} %")