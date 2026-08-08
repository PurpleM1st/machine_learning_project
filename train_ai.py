import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

from database import get_data

# ---------------------------------------------------------------------
# 1. Custom Dataset Wrapper for NumPy Arrays
# ---------------------------------------------------------------------
class AnimalDataset(Dataset):
    def __init__(self, images_array, labels_array, transform=None):
        self.images = images_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Ensure image is uint8 PIL Image for torchvision transforms
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = Image.fromarray(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label

# ---------------------------------------------------------------------
# 2. Fast Label Derivation
# ---------------------------------------------------------------------
def derive_labels_fast(split_name):
    path = f"archive/animals/{split_name}"
    labels = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    folders = sorted([f.name for f in os.scandir(path) if f.is_dir()])
    class_map = {folder_name: idx for idx, folder_name in enumerate(folders)}

    for folder in folders:
        sub_path = os.path.join(path, folder)
        class_id = class_map[folder]
        for entry in os.scandir(sub_path):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts:
                labels.append(class_id)

    return np.array(labels, dtype=np.int64), len(folders)

# ---------------------------------------------------------------------
# 3. Model Definition
# ---------------------------------------------------------------------
def build_model(num_classes, pretrained=True):
    # Load backbone pre-trained on ImageNet
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Fine-tuning: Replace the final classification head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

# ---------------------------------------------------------------------
# 4. Main Pipeline (Training via Gradient Descent)
# ---------------------------------------------------------------------
def train_and_save():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== Step 1: Loading Data & Deriving Labels ===")
    train_x = get_data("train_x.npy")
    val_x = get_data("val_x.npy")

    y_train, num_classes = derive_labels_fast("train")
    y_val, _ = derive_labels_fast("val")

    y_train = y_train[:len(train_x)]
    y_val = y_val[:len(val_x)]

    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)} | Classes: {num_classes}")

    # Data Transforms: Augmentation for Train, Standard Normalization for Val
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AnimalDataset(train_x, y_train, transform=train_transforms)
    val_dataset = AnimalDataset(val_x, y_val, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print("\n=== Step 2: Initializing ResNet-18 Model ===")
    model = build_model(num_classes=num_classes, pretrained=True).to(device)

    # Loss function and Gradient Descent Optimizer (AdamW)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    epochs = 10
    best_val_acc = 0.0

    print("\n=== Step 3: Training Loop ===")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()            # Clear gradients
            outputs = model(images)          # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            loss.backward()                  # Backward pass (gradient computation)
            optimizer.step()                 # Gradient descent step

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_acc = (correct / total) * 100
        train_loss = running_loss / total

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = (val_correct / val_total) * 100
        epoch_val_loss = val_loss / val_total

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "num_classes": num_classes
            }
            torch.save(checkpoint, "models/best_resnet18.pt")

    print(f"\n>>> Best Validation Accuracy Achieved: {best_val_acc:.2f}% <<<")
    print("Saved optimal model checkpoint to 'models/best_resnet18.pt'.")

if __name__ == "__main__":
    train_and_save()