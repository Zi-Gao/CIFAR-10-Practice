import numpy as np
import pandas as pd
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def process_images(image_dir, ids, output_path, desc):
    """Reads images and saves them as a uint8 npy array."""
    images = []
    print(f"Processing {desc}...")
    for idx in tqdm(ids):
        img_path = os.path.join(image_dir, f"{idx}.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    images = np.array(images, dtype=np.uint8)
    # Change shape to (N, C, H, W) for PyTorch compatibility
    images = images.transpose(0, 3, 1, 2)
    np.save(output_path, images)
    print(f"Saved {desc} to {output_path}, shape: {images.shape}")

def main():
    data_dir = "cifar-10-ng/data"
    train_labels_path = os.path.join(data_dir, "trainLabels.csv")
    
    # Load labels
    df = pd.read_csv(train_labels_path)
    
    # Map labels to integers
    all_labels = sorted(df['label'].unique())
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    print(f"Label mapping: {label_to_id}")
    
    # Split training data into train (90%) and val (10%)
    train_idx, val_idx = train_test_split(
        df['id'].values, 
        test_size=0.1, 
        random_state=42, 
        stratify=df['label'].values
    )
    
    # Get labels for the splits
    train_labels = df.set_index('id').loc[train_idx, 'label'].map(label_to_id).values.astype(np.uint8)
    val_labels = df.set_index('id').loc[val_idx, 'label'].map(label_to_id).values.astype(np.uint8)
    
    # Process and save training/val images
    train_dir = os.path.join(data_dir, "train")
    process_images(train_dir, train_idx, os.path.join(data_dir, "train_images.npy"), "Train Images")
    np.save(os.path.join(data_dir, "train_labels.npy"), train_labels)
    
    process_images(train_dir, val_idx, os.path.join(data_dir, "val_images.npy"), "Val Images")
    np.save(os.path.join(data_dir, "val_labels.npy"), val_labels)
    
    # Process test images
    test_dir = os.path.join(data_dir, "test")
    test_ids = range(1, 300001) # Kaggle test set has 300,000 images
    process_images(test_dir, test_ids, os.path.join(data_dir, "test_images.npy"), "Test Images")

if __name__ == "__main__":
    main()
