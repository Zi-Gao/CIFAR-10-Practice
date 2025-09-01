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
# from utils.t import Muon

import numpy as np
from tqdm import tqdm,trange
from torch.utils.tensorboard import SummaryWriter
import torchvision

from utils import read_data
from utils.resnet import resnet18
# from torchvision.models.resnet import resnet18
from utils.normalize import normalize

from collections import deque


torch.set_default_dtype(torch.float32)
torch.autograd.set_detect_anomaly(False)


class CIFAR10_dataset(Dataset):
    def __init__(self,images,labels,transform=None):
        self.images=images
        self.labels=labels
        if transform:
            self.transform=transforms.Compose([
                transform,
                normalize
            ])
        else: self.transform=normalize
    def set_transform(self,transform):
        if transform:
            self.transform=transforms.Compose([
                transform,
                normalize
            ])
        else: self.transform=normalize
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        image=self.images[index]
        label=self.labels[index]
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
        writer.add_scalars('Loss', {f"{self.name}":self.loss/self.total}, epoch)
        writer.add_scalars('Acc', {f"{self.name}":self.acc/self.total}, epoch)

class CheckpointManager:
    """
    一个 checkpoint 管理器，可以：
    1. 只保存最近的 k 个 checkpoint。
    2. 自动保存和更新一个最佳 checkpoint。
    """
    def __init__(self, checkpoint_dir, max_to_keep=3, best_metric_name='val_loss', mode='min'):
        """
        Args:
            checkpoint_dir (str): 保存 checkpoint 的目录。
            max_to_keep (int): 最多保留的近期 checkpoint 数量。
            best_metric_name (str): 用于判断最佳模型的指标名称。
            mode (str): 'min' 或 'max'。'min' 表示指标越小越好。
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_to_keep = max_to_keep
        self.best_metric_name = best_metric_name
        self.mode = mode
        
        # 初始化最佳指标
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        
        # 用于追踪最近的 checkpoint
        self.recent_checkpoints = deque()
        
        # 确保目录存在
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        tqdm.write(f"CheckpointManager initialized. Tracking '{best_metric_name}' in '{mode}' mode.")

    def save(self, state, filename_prefix='ckpt'):
        """
        保存一个新的 checkpoint，管理最近列表，并更新最佳模型。

        Args:
            state (dict): 要保存的状态字典。必须包含 epoch 和 best_metric_name 对应的指标。
            filename_prefix (str): checkpoint 文件名的前缀。
        """
        # 1. 从 state 中获取 epoch 和当前指标
        epoch = state.get('epoch')
        current_metric = state.get(self.best_metric_name)
        
        if epoch is None or current_metric is None:
            raise ValueError(f"State dictionary must contain 'epoch' and '{self.best_metric_name}'")

        # 2. 构造文件名并保存 "最近" checkpoint
        filename = f'{filename_prefix}-epoch={epoch:03d}-{self.best_metric_name}={current_metric:.4f}.pth.tar'
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(state, filepath)
        tqdm.write(f"Saved recent checkpoint: {filename}")
        
        # 3. 管理最近 checkpoint 队列
        self.recent_checkpoints.append(filepath)
        if len(self.recent_checkpoints) > self.max_to_keep:
            oldest_checkpoint = self.recent_checkpoints.popleft()
            try:
                os.remove(oldest_checkpoint)
                tqdm.write(f"Removed old checkpoint: {os.path.basename(oldest_checkpoint)}")
            except OSError as e:
                tqdm.write(f"Error removing old checkpoint {oldest_checkpoint}: {e}")

        # 4. 检查并保存 "最佳" checkpoint
        is_best = (self.mode == 'min' and current_metric < self.best_metric) or \
                  (self.mode == 'max' and current_metric > self.best_metric)
                  
        if is_best:
            self.best_metric = current_metric
            best_filepath = os.path.join(self.checkpoint_dir, 'best.pth.tar')
            shutil.copyfile(filepath, best_filepath)
            tqdm.write(f"🎉 New best model found and saved to 'best.pth.tar' with {self.best_metric_name}: {self.best_metric:.4f}")

    def load(self, model, optimizer, filepath):
        """加载一个 checkpoint。"""
        # (这个加载逻辑可以根据需要添加)
        if not os.path.exists(filepath):
            tqdm.write(f"Checkpoint file not found at '{filepath}'.")
            return None
        
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        tqdm.write(f"Loaded checkpoint from '{filepath}'.")
        return checkpoint

class MultiOptimizer(optim.Optimizer):
    def __init__(self, optimizers):
        self.optimizers = optimizers if isinstance(optimizers, list) else list(optimizers.values())
        self.param_groups = []
        for optimizer in self.optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def step(self, closure=None):
        raise NotImplementedError("MultiOptimizer.step() should not be called.")

    def zero_grad(self, set_to_none: bool = False):
        raise NotImplementedError("MultiOptimizer.zero_grad() should not be called.")

def get_augment_transform(p):
    return transforms.Compose([
        transforms.RandomErasing(p=1, scale=(0.25*p, 0.5*p), ratio=(0.3*p, 3.3*p), value=0),
        transforms.RandomCrop(32,padding=int(4*p)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=int(9*p)),
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

    batch_size=256
    train_dataset=CIFAR10_dataset(train_images,train_labels,get_augment_transform(1))
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1,persistent_workers=True)

    vaild_batch_size=512
    test_dataset=CIFAR10_dataset(test_images,test_labels)
    test_loader=DataLoader(test_dataset,batch_size=vaild_batch_size,shuffle=False,num_workers=1,persistent_workers=True)

    # ==============================================================================
    # 3. 模型、损失函数、优化器设置
    # ==============================================================================
    print("\n3. Setting up model and training components...")
    model = resnet18(num_classes=NUM_CLASS)

    model = model.to(device)
    # checkpoint = torch.load('./run/resnet18/3/ckpt/best.pth.tar', map_location=device)
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

    num_epochs=100

    peak_lr_muon = 0.02
    floor_lr_muon = 0.02
    patience_muon=10
    decay_factor_muon=0.7
    warmup_epochs_muon=1
    cosine_end_epoch_muon=2

    optimizer_muon = SingleDeviceMuon(muon_params,lr=peak_lr_muon,ns_steps=2)

    warmup_scheduler_muon = LinearLR(optimizer_muon, start_factor=0.01, total_iters=warmup_epochs_muon)
    cosine_scheduler_muon = CosineAnnealingLR(optimizer_muon, 
                                        T_max=(cosine_end_epoch_muon - warmup_epochs_muon), 
                                        eta_min=floor_lr_muon)
    
    plateau_scheduler_muon = ReduceLROnPlateau(optimizer_muon, 
                                        mode='min', 
                                        factor=decay_factor_muon, 
                                        patience=patience_muon)

    sequential_scheduler_muon = SequentialLR(optimizer_muon, 
                                    schedulers=[warmup_scheduler_muon, cosine_scheduler_muon], 
                                    milestones=[warmup_epochs_muon])


    peak_lr_adam = 0.005
    floor_lr_adam = 0.005
    patience_adam=patience_muon
    decay_factor_adam=decay_factor_muon
    warmup_epochs_adam=warmup_epochs_muon
    cosine_end_epoch_adam=cosine_end_epoch_muon

    optimizer_adam = optim.AdamW(other_params,lr=peak_lr_adam,weight_decay=0.005)

    warmup_scheduler_adam = LinearLR(optimizer_adam, start_factor=0.01, total_iters=warmup_epochs_adam)
    cosine_scheduler_adam = CosineAnnealingLR(optimizer_adam, 
                                        T_max=(cosine_end_epoch_adam - warmup_epochs_adam), 
                                        eta_min=floor_lr_adam)
    
    plateau_scheduler_adam = ReduceLROnPlateau(optimizer_adam, 
                                        mode='min', 
                                        factor=decay_factor_adam, 
                                        patience=patience_adam)

    sequential_scheduler_adam = SequentialLR(optimizer_adam, 
                                    schedulers=[warmup_scheduler_adam, cosine_scheduler_adam], 
                                    milestones=[warmup_epochs_adam])
    
    optimizer_list=[optimizer_muon,optimizer_adam]

    print(f"Muon Optimizer is handling {len(muon_params)} parameter tensors.")
    print(f"Adam Optimizer is handling {len(other_params)} parameter tensors.")

    print("Setup complete.")

    # ==============================================================================
    # 4. 训练循环
    # ==============================================================================
    run_path='run/resnet18/4'
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


    from tensorboard.program import TensorBoard
    tb = TensorBoard()
    tb.configure(argv=[None, '--logdir', writer_path, '--port', str(6006)])
    url = tb.launch() # 这会启动一个后台线程来运行 TensorBoard
    print(f"TensorBoard started at {url}")

    checkpoint_manager=CheckpointManager(checkpoint_path,16)

    print("\n4. Starting training ...")

    cnt=0
    loss_min=np.inf
    current_p=1
    cnt_bad=0

    for epoch in trange(num_epochs):
        model.train()
        train_evaler=Evaler('train')
        progress_bar = tqdm(train_loader, desc=f"Epoch(Train) {epoch+1}/{num_epochs}",leave=False)


        # @torch.compile
        # def train(images,labels):
        #     images=images.to(device)
        #     labels=labels.to(device)

        #     for optimizer in optimizer_list:
        #         optimizer.zero_grad(set_to_none=True)

        #     outputs=model(images)
        #     loss=criterion(outputs,labels)

        #     loss.backward()
        #     for optimizer in optimizer_list:
        #         optimizer.step()
            
        #     train_evaler.upd(outputs,labels,loss)

        for images,labels in progress_bar:
            # train(images,labels)
            images=images.to(device)
            labels=labels.to(device)

            for optimizer in optimizer_list:
                optimizer.zero_grad()

            outputs=model(images)
            loss=criterion(outputs,labels)

            loss.backward()
            for optimizer in optimizer_list:
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

        tqdm.write("train:"+str(train_evaler.result()))
        tqdm.write("vaild:"+str(vaild_evaler.result()))
        train_evaler.upd_board(writer,epoch)
        vaild_evaler.upd_board(writer,epoch)

        val_loss=vaild_evaler.result()["loss"]

        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        checkpoint_manager.save(state)

        # if val_loss>loss_min:
        #     cnt_bad+=1
        # if val_loss<loss_min:
        #     cnt_bad=0
        # if cnt_bad==20

        # val_loss = 0.5 / (epoch + 1)
        # if epoch > 210 and epoch % 15 > 10: val_loss = 0.1/(epoch+1)

        if epoch < cosine_end_epoch_muon:
            sequential_scheduler_muon.step()
            sequential_scheduler_adam.step()
        else:
            plateau_scheduler_muon.step(val_loss)
            plateau_scheduler_adam.step(val_loss)
            if val_loss<loss_min:
                cnt=0
                loss_min=val_loss
            else: cnt+=1
            if cnt==10:
                current_p*=0.5
                cnt=0
                train_dataset.set_transform(get_augment_transform(current_p))

        writer.add_scalars('Learning rate',{"Muon":optimizer_muon.param_groups[0]['lr']},epoch)
        writer.add_scalars('Learning rate',{"Adam":optimizer_adam.param_groups[0]['lr']},epoch)
        
        writer.add_scalar('Vaild loss',val_loss,epoch)
        writer.add_scalar('Augment',current_p,epoch)
        
        

        original_images = train_images[:8]
        augmented_images = torch.stack([get_augment_transform(current_p)(img) for img in original_images])
        
        comparison_batch = torch.cat([original_images, augmented_images]).float()
        grid = torchvision.utils.make_grid(comparison_batch, nrow=8, normalize=True)
        writer.add_image('Original (top) vs. Augmented (bottom)', grid, epoch)


if __name__ == '__main__':

    main()