import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm,trange
from torch.utils.tensorboard import SummaryWriter

from utils import read_data
from utils.resnet import resnet18
from utils.normalize import normalize

torch.set_default_dtype(torch.float32)

class MyDataset(Dataset):
    def __init__(self,images,labels,transform=None):
        self.images=images
        self.labels=labels
        self.transform=transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image=self.images[index]
        label=self.labels[index]
        if self.transform:
            image=self.transform(image)
        return image,label

class Evaler:
    def __init__(self,name):
        self.loss=0
        self.acc=0
        self.total=0
        self.name=name

    def upd(self,outputs,labels,loss):
        batch_size=labels.size(0)
        self.loss+=loss.item()*batch_size
        self.total+=batch_size
        
        _, predicted = torch.max(outputs, 1)
        self.acc+=(predicted == labels).sum().item()
    
    def result(self):
        return {
            "loss":self.loss/self.total,
            "acc":self.acc/self.total,
            "total":self.total,
        }
    
    def upd_board(self,writer,epoch):
        writer.add_scalar(f'Loss/{self.name}', self.loss/self.total, epoch)
        writer.add_scalar(f'Acc/{self.name}', self.acc/self.total, epoch)
        

def main():

    # ==============================================================================
    # 0. 加载数据
    # ==============================================================================
    print("0. Loading data...")
    NUM_CLASS=10
    train_pic = read_data.get_train_pic_train()
    train_label = read_data.get_train_label_train()
    test_pic = read_data.get_train_pic_test()
    test_label = read_data.get_train_label_test()

    print(f"Train pictures shape: {train_pic.shape}, dtype: {train_pic.dtype}")
    print(f"Train labels shape: {train_label.shape}")
    print(f"Test pictures shape: {test_pic.shape}, dtype: {test_pic.dtype}")
    print(f"Test labels shape: {test_label.shape}")

    # ==============================================================================
    # 1. 设备配置
    # ==============================================================================
    print("\n1. Configuring device...")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    else:
        device = torch.device("cpu")
        print("MPS not available, using CPU instead.")

    # ==============================================================================
    # 2. 数据准备
    # ==============================================================================


    print("\n2. Preparing data...")
    train_images=torch.from_numpy(train_pic)
    train_labels=torch.from_numpy(train_label).long()
    test_images=torch.from_numpy(test_pic)
    test_labels=torch.from_numpy(test_label).long()

    print("\n2.5. Setting up data augmentation and conversion pipeline...")

    train_transform=transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        normalize
    ])
    test_transform=normalize


    batch_size=256
    train_dataset=MyDataset(train_images,train_labels,train_transform)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1,persistent_workers=True)

    vaild_batch_size=512
    test_dataset=MyDataset(test_images,test_labels,test_transform)
    test_loader=DataLoader(test_dataset,batch_size=vaild_batch_size,shuffle=False,num_workers=1,persistent_workers=True)

    # ==============================================================================
    # 3. 模型、损失函数、优化器设置
    # ==============================================================================
    print("\n3. Setting up model and training components...")
    model = resnet18(num_classes=NUM_CLASS).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Setup complete.")

    # ==============================================================================
    # 4. 训练循环
    # ==============================================================================

    writer = SummaryWriter('CIFAR-10')

    print("\n4. Starting training ...")

    num_epochs=20
    for epoch in trange(num_epochs):
        model.train()
        train_evaler=Evaler('train')
        progress_bar = tqdm(train_loader, desc=f"Epoch(Train) {epoch+1}/{num_epochs}",leave=False)
        for images,labels in progress_bar:
            # images=train_transform(images)

            images=images.to(device)
            labels=labels.to(device)
            
            optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs,labels)

            loss.backward()
            optimizer.step()

            train_evaler.upd(outputs,labels,loss)


        model.eval()
        vaild_evaler=Evaler('vaild')
        progress_bar = tqdm(test_loader, desc=f"Epoch(Vaild) {epoch+1}/{num_epochs}",leave=False)
        for images,labels in progress_bar:
            # images=test_transform(images)

            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            loss=criterion(outputs,labels)
            
            vaild_evaler.upd(outputs,labels,loss)

        tqdm.write(f"epoch:{epoch+1}/num_epochs")

        tqdm.write("train:"+str(train_evaler.result()))
        tqdm.write("vaild:"+str(vaild_evaler.result()))
        train_evaler.upd_board(writer,epoch)
        vaild_evaler.upd_board(writer,epoch)
        

if __name__ == '__main__':
    main()