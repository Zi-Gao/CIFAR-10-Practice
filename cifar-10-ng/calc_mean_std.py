import numpy as np
import os

def calc_stats(file_path, desc):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None, None

    print(f"Loading {file_path}...")
    # Load as uint8
    images = np.load(file_path)
    
    # Convert to float and scale to [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # Calculate mean and std per channel
    # images shape: (N, C, H, W)
    mean = np.mean(images, axis=(0, 2, 3))
    std = np.std(images, axis=(0, 2, 3))
    
    print(f"{desc} Statistics:")
    print(f"  Mean: {mean}")
    print(f"  Std:  {std}")
    print(f"  Formatted: MEAN = ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}), STD = ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")
    return mean, std

def main():
    data_dir = "cifar-10-ng/data"
    
    datasets = [
        ("train_images.npy", "Train"),
        ("val_images.npy", "Val"),
        ("test_images.npy", "Test")
    ]
    
    for filename, desc in datasets:
        calc_stats(os.path.join(data_dir, filename), desc)
        print("-" * 50)

if __name__ == "__main__":
    main()
