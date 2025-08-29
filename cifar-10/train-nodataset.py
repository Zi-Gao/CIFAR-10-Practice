import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm,trange

from utils import read_data
from utils.resnet import resnet18

# ==============================================================================
# 0. 加载数据
# ==============================================================================
print("0. Loading data...")
NUM_CLASS = 10
train_pic = read_data.get_train_pic_train()
train_label = read_data.get_train_label_train()
test_pic = read_data.get_train_pic_test()
test_label = read_data.get_train_label_test()

print(f"Train pictures shape: {train_pic.shape}")
print(f"Train labels shape: {train_label.shape}")
print(f"Test pictures shape: {test_pic.shape}")
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
torch.manual_seed(42)
torch.set_default_dtype(torch.float32)

# ==============================================================================
# 2. 数据准备和预处理
# ==============================================================================
print("\n2. Preparing data...")

# ==============================================================================
# 3. 模型、损失函数、优化器和混合精度设置
# ==============================================================================
print("\n3. Setting up model and training components...")
model = resnet18(num_classes=NUM_CLASS).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

batch_size = 64

# ==============================================================================
# 4. 训练循环
# ==============================================================================
print("\n4. Starting training...")
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    def split_batch(batch_size,X,y):
        cnt = (len(X)+batch_size-1)//batch_size
        progress_bar = tqdm(range(cnt), desc=f"Epoch {epoch+1}/{num_epochs}")

        X_tensor=torch.from_numpy(X).to(device)
        y_tensor=torch.from_numpy(y).to(device)

        for i in progress_bar:
            yield X_tensor[i*batch_size:(i+1)*batch_size],y_tensor[i*batch_size:(i+1)*batch_size],progress_bar

    for images,labels,progress_bar in split_batch(batch_size,train_pic,train_label):
        optimizer.zero_grad()
        
        # with torch.autocast(device_type="mps", dtype=torch.float16):
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*len(images)
        
        progress_bar.set_postfix(loss=loss.item())


    epoch_loss = running_loss / len(train_pic)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss:.4f}")

# # ==============================================================================
# # 5. 评估模型
# # ==============================================================================
# print("\n5. Evaluating model on test data...")
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in tqdm(test_loader, desc="Evaluating"):
#         images = images.to(device)
#         labels = labels.to(device)
        
#         with torch.autocast(device_type="mps", dtype=torch.float16):
#             outputs = model(images)
            
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f"Accuracy on the test set: {accuracy:.2f} %")