import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from datetime import datetime
import re
import glob

from model import resnet18
from utils.data_utils import CIFAR10NpyDataset, get_train_transform, get_val_transform
from utils.inference import EMA, predict, predict_tta

# Training Hyperparameters
BATCH_SIZE = 128
EPOCHS = 40
LR_MUON = 0.02
MOMENTUM_MUON = 0.95
LR_1D = 3e-4
WD_1D = 0.01

def find_latest_checkpoint(resume_path):
    if os.path.isfile(resume_path):
        return resume_path
    
    # If it's a directory, search for the latest epoch checkpoint
    checkpoint_files = glob.glob(os.path.join(resume_path, "resnet18_epoch*.pth"))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the max
    epochs = []
    for f in checkpoint_files:
        match = re.search(r"resnet18_epoch(\d+)\.pth", f)
        if match:
            epochs.append((int(match.group(1)), f))
    
    if not epochs:
        return None
        
    # Sort by epoch number and return the latest
    epochs.sort(key=lambda x: x[0], reverse=True)
    return epochs[0][1]

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 NG Training")
    parser.add_argument("--remark", type=str, default="no remark", help="Remark for this experiment")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--opt1d", type=str, default="adamw", choices=["adamw", "sgd"], help="Optimizer for 1D parameters")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA for validation")
    parser.add_argument("--no_ema", action="store_false", dest="use_ema", help="Disable EMA")
    parser.add_argument("--ema_buffers", action="store_true", default=False, help="Also EMA BatchNorm buffers")
    parser.add_argument("--resume", type=str, default=None, help="Path to a run directory or a specific checkpoint to resume from")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    start_epoch = 0
    best_acc = 0
    run_dir = None
    checkpoint_path = None

    if args.resume:
        checkpoint_path = find_latest_checkpoint(args.resume)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            # If we are resuming from a directory, we might want to continue logging in the same dir
            # but usually it's safer to create a new one unless specified. 
            # For simplicity, let's just create a new run dir for the resume session.
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            print(f"No checkpoint found at {args.resume}. Starting from scratch.")
            args.resume = None

    # Create run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = timestamp + (f"_resume" if args.resume else "")
    run_dir = os.path.join("cifar-10-ng/runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Save remark to file
    with open(os.path.join(run_dir, "remark.txt"), "w") as f:
        f.write(f"Remark: {args.remark}\n")
        f.write(f"Optimizer 1D: {args.opt1d}\n")
        if args.resume:
            f.write(f"Resumed from: {args.resume}\n")
    
    print(f"Logging to: {run_dir}")
    print(f"Remark: {args.remark}")
    print(f"1D Optimizer: {args.opt1d}")

    # Datasets & Dataloaders
    data_dir = "cifar-10-ng/data"
    train_dataset = CIFAR10NpyDataset(
        os.path.join(data_dir, "train_images.npy"),
        os.path.join(data_dir, "train_labels.npy"),
        transform=get_train_transform()
    )
    val_dataset = CIFAR10NpyDataset(
        os.path.join(data_dir, "val_images.npy"),
        os.path.join(data_dir, "val_labels.npy"),
        transform=get_val_transform()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model & EMA
    model = resnet18(num_classes=10).to(device)
    ema = EMA(model, decay=0.999, ema_buffers=args.ema_buffers)

    if args.resume and checkpoint_path:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback for old style checkpoints that only saved state_dict
            model.load_state_dict(checkpoint)

    # Separate parameters for Muon and 1D Optimizer
    muon_params = []
    params_1d = []
    print("\n--- Optimizer Parameter Assignment ---")
    print("Muon (2D weights):")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2: # 2D weights (Conv, Linear)
            muon_params.append(param)
            print(f"  - {name:30} | Shape: {str(list(param.shape)):20}")
        else: # 1D params (Bias, BN)
            params_1d.append(param)
    
    print(f"\n{args.opt1d.upper()} (1D params):")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2:
            print(f"  - {name:30} | Shape: {str(list(param.shape)):20}")
    print("--------------------------------------\n")
    
    # Using official torch.optim.Muon (requires 2D parameters)
    optimizer_muon = optim.Muon(muon_params, lr=LR_MUON, momentum=MOMENTUM_MUON, nesterov=True, weight_decay=0)
    
    if args.opt1d == "adamw":
        optimizer_1d = optim.AdamW(params_1d, lr=LR_1D, weight_decay=WD_1D)
    else: # sgd
        optimizer_1d = optim.SGD(params_1d, lr=LR_1D, momentum=0.9, weight_decay=WD_1D)

    # Schedulers
    total_steps = len(train_loader) * args.epochs
    scheduler_muon = optim.lr_scheduler.OneCycleLR(optimizer_muon, max_lr=LR_MUON, total_steps=total_steps)
    scheduler_1d = optim.lr_scheduler.OneCycleLR(optimizer_1d, max_lr=optimizer_1d.param_groups[0]['lr'], total_steps=total_steps)

    if args.resume and checkpoint_path and 'optimizer_muon_state_dict' in checkpoint:
        optimizer_muon.load_state_dict(checkpoint['optimizer_muon_state_dict'])
        optimizer_1d.load_state_dict(checkpoint['optimizer_1d_state_dict'])
        scheduler_muon.load_state_dict(checkpoint['scheduler_muon_state_dict'])
        scheduler_1d.load_state_dict(checkpoint['scheduler_1d_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)
        if args.use_ema:
            if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
                ema.load_state_dict(checkpoint['ema_state_dict'])
            else:
                print("Warning: No EMA state found in checkpoint. Re-initializing from current model weights.")
                ema.initialize_shadow()
        print(f"Resuming from epoch {start_epoch}")

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Logging
    writer = SummaryWriter(run_dir)

    # Training Loop
    pbar_total = tqdm(range(start_epoch, args.epochs), desc="Total Training Progress")
    for epoch in pbar_total:
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for images, labels in pbar_train:
            images, labels = images.to(device), labels.to(device)
            
            optimizer_muon.zero_grad()
            optimizer_1d.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer_muon.step()
            optimizer_1d.step()
            
            scheduler_muon.step()
            scheduler_1d.step()
            
            if args.use_ema:
                ema.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar_train.set_postfix({"Loss": f"{train_loss/(pbar_train.n+1):.4f}", "Acc": f"{100.*correct/total:.2f}%"})

        # Validation
        train_acc = 100. * correct / total
        train_loss_avg = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        correct_tta = 0
        
        # Apply EMA weights for validation
        if args.use_ema:
            ema.apply_shadow()
        
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in pbar_val:
                images, labels = images.to(device), labels.to(device)
                
                # Standard Inference
                outputs = predict(model, images, device)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # TTA Inference
                outputs_tta = predict_tta(model, images, device)
                _, predicted_tta = outputs_tta.max(1)
                correct_tta += predicted_tta.eq(labels).sum().item()
                
                pbar_val.set_postfix({"Loss": f"{val_loss/(pbar_val.n+1):.4f}", "Acc": f"{100.*correct/total:.2f}%"})

        val_acc = 100. * correct / total
        val_acc_tta = 100. * correct_tta / total
        
        writer.add_scalar("Loss/Train", train_loss_avg, epoch)
        writer.add_scalar("Loss/Val", val_loss/len(val_loader), epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("Accuracy/Val_TTA", val_acc_tta, epoch)
        
        # Restore original weights
        if args.use_ema:
            ema.restore()
        
        # Save Checkpoint Helper
        def save_checkpoint(filename):
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                'optimizer_1d_state_dict': optimizer_1d.state_dict(),
                'scheduler_muon_state_dict': scheduler_muon.state_dict(),
                'scheduler_1d_state_dict': scheduler_1d.state_dict(),
                'best_acc': best_acc,
                'ema_state_dict': ema.state_dict() if args.use_ema else None,
            }
            torch.save(ckpt, os.path.join(run_dir, filename))

        # Save Best and Periodic Checkpoints
        if val_acc_tta > best_acc:
            best_acc = val_acc_tta
            save_checkpoint("best_model.pth")
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(f"resnet18_epoch{epoch+1}.pth")
        
        # Update Total Progress Bar with stats
        pbar_total.set_postfix({"Best TTA Acc": f"{best_acc:.2f}%", "Last Val Acc": f"{val_acc:.2f}%"})

    writer.close()
    print(f"\nTraining complete. Best Val TTA Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
