import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from sklearn.manifold import TSNE

try:
    from database import get_data
except ImportError:
    raise ImportError("This script must be run from the directory containing 'database.py' or 'database.py' must be in your python path.")

class AnimalDataset(Dataset):
    def __init__(self, images_array, labels_array, transform=None):
        self.images = images_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = Image.fromarray(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label

# --- Helper to Get Class Names ---
def get_class_names(split_name="val"):
    """Gets sorted class folder names to map IDs to labels."""
    path = f"archive/animals/{split_name}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")
    folders = sorted([f.name for f in os.scandir(path) if f.is_dir()])
    return folders

# --- Feature Extraction Model Builder ---
def build_feature_extractor(model_path, device):
    """
    Loads the trained model and modifies it to output 
    raw features (the 512D vector) instead of classifications.
    """
    print(f"\n[1/5] Loading saved weights from '{model_path}'...")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint["num_classes"]

    # Reconstruct base model (weights=None because we load our own)
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval() # CRITICAL: Set to evaluation mode
    
    return feature_extractor

# The main pipeline funtion that calls the others
def main():
    model_path = "models/best_resnet18.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at '{model_path}'. Check path.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Modified Model
    model = build_feature_extractor(model_path, device)
    
    try:
        class_names = get_class_names()
    except Exception as e:
        print(f"Warning: Could not get class names automatically: {e}")
        class_names = None

    # 3. Load Validation Data Arrays (using your get_data helper)
    print("[2/5] Loading validation dataset from cache...")
    try:
        val_x = get_data("val_x.npy")
        from main import derive_labels_fast
        y_val, _ = derive_labels_fast("val")
        y_val = y_val[:len(val_x)] # Ensure alignment
        
    except Exception as e:
        raise ImportError(f"Failed to load data arrays or derive labels: {e}. "
                          f"Ensure 'database.py' and 'main.py' are in place.")

    # Apply same evaluation transforms as main script
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = AnimalDataset(val_x, y_val, transform=eval_transforms)
    # Batch size can be larger as we are just inferencing
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 4. Feature Extraction Loop
    print("[3/5] Extracting 512-D feature vectors (inference)...")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            
            # Forward pass: Output is (Batch, 512, 1, 1)
            features = model(images)
            
            # Flatten to (Batch, 512)
            features = features.view(features.size(0), -1)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    # Concatenate into large numpy arrays
    X = np.concatenate(all_features, axis=0) # Shape (N, 512)
    y = np.concatenate(all_labels, axis=0)    # Shape (N,)

    print(f"Extracted {X.shape[0]} feature vectors of dimension {X.shape[1]}.")

    # 5. Run t-SNE Dimensionality Reduction
    print("[4/5] Running t-SNE reduction (this may take a few minutes)...")
    # Perplexity 30-50 is a good starting point. Lower 'n_iter' if too slow.
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42, verbose=1)
    X_embedded = tsne.fit_transform(X) # Shape (N, 2)

    # 6. Plotting
    print("[5/5] Generating scatter plot...")
    plt.figure(figsize=(12, 10))
    plt.style.use('seaborn-v0_8-whitegrid') # Cleaner look for scatter

    num_classes = len(class_names) if class_names else len(np.unique(y))
    cmap = plt.cm.get_cmap('tab10', num_classes) # Qualitative colormap for categories

    # Plot each class individually to build the legend
    for i in range(num_classes):
        indices = np.where(y == i)
        label_text = class_names[i] if class_names else f"Class {i}"
        
        plt.scatter(X_embedded[indices, 0], 
                    X_embedded[indices, 1], 
                    c=[cmap(i)], 
                    label=label_text, 
                    alpha=0.6, 
                    s=20, # Point size
                    edgecolors='none')

    plt.legend(markerscale=2, loc='best', fontsize='medium', frameon=True)
    plt.title(f"t-SNE Visualization of ResNet-18 Latent Space (Validation Set)\n"
              f"Model: {os.path.basename(model_path)}", fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    save_path = "resnet18_tsne_clusters.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved t-SNE cluster chart to '{save_path}'")
    
    plt.show()

if __name__ == "__main__":
    main()