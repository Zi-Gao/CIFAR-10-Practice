import os
import math
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

from torch.optim.lr_scheduler import *

# from muon import SingleDeviceMuon
from utils.Moonshot_muon import Muon

import numpy as np
from tqdm import tqdm,trange
from torch.utils.tensorboard import SummaryWriter
import torchvision

from utils import read_data
from utils.resnet import resnet18
# from torchvision.models.resnet import resnet18
from utils.normalize import normalize

from collections import deque

from utils.utils import *


torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(False)

def get_augment_transform(p):
    return transforms.Compose([
        transforms.RandomErasing(p=1, scale=(0.25*p, 0.5*p), ratio=(0.3*p, 3.3*p), value=0),
        transforms.RandomCrop(32,padding=4),
        # transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandAugment(num_ops=2, magnitude=int(4*p)),
        v2.AugMix()
    ])

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

    # augment_transform=get_augment_transform(1)

    batch_size=512
    train_dataset=CIFAR10_dataset(train_images,train_labels,get_augment_transform(1))
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=2,persistent_workers=True)

    vaild_batch_size=512
    test_dataset=CIFAR10_dataset(test_images,test_labels)
    test_loader=DataLoader(test_dataset,batch_size=vaild_batch_size,shuffle=False,num_workers=2,persistent_workers=True)

    # ==============================================================================
    # 3. 模型、损失函数、优化器设置
    # ==============================================================================
    print("\n3. Setting up model and training components...")
    model = resnet18(num_classes=NUM_CLASS)

    model = model.to(device)
    # checkpoint = torch.load('./run/resnet18/5/ckpt/best.pth.tar', map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])

    # print(type(model))

    criterion = nn.CrossEntropyLoss()

    muon_params = []
    other_params = []


    print("--- Separating Parameters ---")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "conv1.weight" != name and "fc.weight" != name:
            print(f"✅ Assigning '{name}' (shape: {param.shape}) to Muon Optimizer.")
            muon_params.append(param)
        else:
            print(f"⚪️ Assigning '{name}' (shape: {param.shape}) to Adam Optimizer.")
            other_params.append(param)

    total_params = len(list(model.parameters()))
    assigned_params = len(muon_params) + len(other_params)
    assert total_params == assigned_params, "Some parameters were not assigned to an optimizer!"

    num_epochs=300

    optimizer = Muon([
        {'params': muon_params, "use_muon": True, 'lr': 0.02, 'weight_decay': 0.001}, # group 0
        {'params': other_params, "use_muon": False, 'lr': 0.001, 'weight_decay': 0.001} # group 1
    ])

    lr_scheduler = OneCycleLR(optimizer,  total_steps=num_epochs,max_lr=[0.02,0.001])
    # lr_scheduler=ReduceLROnPlateau(optimizer,mode='max',factor=0.5)

    print(f"Muon Optimizer is handling {len(muon_params)} parameter tensors.")
    print(f"Adam Optimizer is handling {len(other_params)} parameter tensors.")

    print("Setup complete.")

    # ==============================================================================
    # 4. 训练循环
    # ==============================================================================
    run_path='run/resnet18/8'
    writer_path=f"{run_path}/tb"
    checkpoint_path=f"{run_path}/ckpt"
    if os.path.exists(run_path):
        print(f"Log directory '{run_path}' already exists. Input yes to remove it.")
        confirm=input()
        if confirm!="yes":
            print("Cancelled.")
            exit(0)
        shutil.rmtree(run_path)
    os.makedirs(run_path)

    writer = SummaryWriter(writer_path)


    # from tensorboard.program import TensorBoard
    # tb = TensorBoard()
    # tb.configure(argv=[None, '--logdir', writer_path, '--port', str(6006)])
    # url = tb.launch() # 这会启动一个后台线程来运行 TensorBoard
    # print(f"TensorBoard started at {url}")

    checkpoint_manager=CheckpointManager(checkpoint_path,16,'val_acc','max')

    print("\n4. Starting training ...")

    cnt=0
    loss_min=np.inf
    acc_max=-np.inf
    current_p=1.0

    cutmix=v2.CutMix(num_classes=NUM_CLASS)

    for epoch in trange(num_epochs):
        model.train()
        train_evaler=Evaler('train')
        progress_bar = tqdm(train_loader, desc=f"Epoch(Train) {epoch+1}/{num_epochs}",leave=False)

        for images,labels in progress_bar:
            images=images.to(device)
            labels=labels.to(device)
            images,labels=cutmix(images,labels)

            # print(labels.shape)

            optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs,labels)

            loss.backward()
            optimizer.step()
            train_evaler.upd(outputs,labels,loss)


        model.eval()
        vaild_evaler=Evaler('vaild')
        progress_bar = tqdm(test_loader, desc=f"Epoch(Vaild) {epoch+1}/{num_epochs}",leave=False)
        with torch.inference_mode():
            for images,labels in progress_bar:

                images=images.to(device)
                labels=labels.to(device)

                outputs=model(images)
                loss=criterion(outputs,labels)
                
                vaild_evaler.upd(outputs,labels,loss)

        tqdm.write(f"epoch:{epoch+1}/num_epochs")



        val_loss=vaild_evaler.result()["loss"]
        val_acc=vaild_evaler.result()["acc"]
        

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        checkpoint_manager.save(state)
        
        lr_scheduler.step()

        # if val_acc>acc_max:
        #     acc_max=val_acc
        #     cnt=0
        # else: cnt+=1
        # if cnt==10:
        #     current_p*=0.8
        #     train_dataset.transform=get_augment_transform(current_p)
        #     cnt=0
        #     tqdm.write(f"Reduce augment p!! now:{current_p}")

        # lr_scheduler.step(val_acc)

        tqdm.write("train:"+str(train_evaler.result()))
        tqdm.write("vaild:"+str(vaild_evaler.result()))
        tqdm.write(f"cnt: {cnt} acc_max:{acc_max}")
        train_evaler.upd_board(writer,epoch)
        vaild_evaler.upd_board(writer,epoch)

        writer.add_scalars('Learning rate',{"Muon":optimizer.param_groups[0]['lr']},epoch)
        writer.add_scalars('Learning rate',{"Adam":optimizer.param_groups[1]['lr']},epoch)
        

        # writer.add_scalar('Vaild loss',val_loss,epoch)
        writer.add_scalar('Augment',current_p,epoch)
        
        original_images = train_images[:8]
        augmented_images = torch.stack([get_augment_transform(current_p)(img) for img in original_images])
        
        comparison_batch = torch.cat([original_images, augmented_images]).float()
        grid = torchvision.utils.make_grid(comparison_batch, nrow=8, normalize=True)
        writer.add_image('Original (top) vs. Augmented (bottom)', grid, epoch)


if __name__ == '__main__':

    main()