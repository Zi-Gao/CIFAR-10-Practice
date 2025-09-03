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
        if labels.ndim==1:
            self.acc+=(predicted == labels).sum().item()
        else:
            _, answer = torch.max(labels, 1)
            self.acc+=(predicted == answer).sum().item()
    
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
