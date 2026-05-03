import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse

from model import resnet18
from utils.data_utils import CIFAR10NpyDataset, get_val_transform
from utils.inference import predict_tta,predict

def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 NG Testing")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for inference")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = resnet18(num_classes=10).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint from: {args.ckpt}")

    # Dataset & Dataloader
    data_dir = "cifar-10-ng/data"
    test_dataset = CIFAR10NpyDataset(
        os.path.join(data_dir, "test_images.npy"),
        transform=get_val_transform() # Use val transform for test
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Class Mapping (alphabetical order as per Kaggle)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Inference
    results = []
    print("Running inference on test set (with TTA)...")
    with torch.no_grad():
        for i, images in enumerate(tqdm(test_loader)):
            outputs = predict_tta(model, images, device)
            _, predicted = outputs.max(1)
            
            # Map IDs to labels
            for p in predicted:
                results.append(classes[p.item()])

    # Save Submission
    submission = pd.DataFrame({
        "id": range(1, len(results) + 1),
        "label": results
    })
    
    # Save in the same directory as checkpoint
    save_path = os.path.join(os.path.dirname(args.ckpt), "submission.csv")
    submission.to_csv(save_path, index=False)
    print(f"Submission saved to: {save_path}")

if __name__ == "__main__":
    main()
