import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

# Import data helper from your existing database.py
from database import get_data

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


def build_model(num_classes):
    model = models.resnet18(weights=None)  
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# Evaluates the statistics of the trained model
def evaluate_dataset(model, data_loader, criterion, device):
    """Calculates overall loss and accuracy for a given dataset loader."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = (correct / total) * 100
    avg_loss = running_loss / total
    return avg_loss, acc


# The main that combines everything, loading, predicting and creating the graphs
def main():
    model_path = "models/best_resnet18.pt"
    os.makedirs("graphs", exist_ok=True)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at '{model_path}'.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Checkpoint
    print(f"\nLoading saved weights from '{model_path}'...")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint["num_classes"]

    # 2. Reconstruct Model
    model = build_model(num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    # 3. Load Data Arrays
    print("\nLoading datasets from cache...")
    train_x = get_data("train_x.npy")
    val_x = get_data("val_x.npy")

    y_train, _ = derive_labels_fast("train")
    y_val, _ = derive_labels_fast("val")

    y_train = y_train[:len(train_x)]
    y_val = y_val[:len(val_x)]

    # Standard evaluation transform (no random flips/rotations needed for evaluation)
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = AnimalDataset(train_x, y_train, transform=eval_transforms)
    val_dataset = AnimalDataset(val_x, y_val, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 4. Evaluate Saved Model Performance
    print("\nEvaluating saved model on Train and Validation sets...")
    train_loss, train_acc = evaluate_dataset(model, train_loader, criterion, device)
    val_loss, val_acc = evaluate_dataset(model, val_loader, criterion, device)

    print("\n=== Saved Model Results ===")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

    # 5. Plot Comparison Chart
    print("\nGenerating evaluation comparison chart...")
    categories = ['Training Set', 'Validation Set']
    losses = [train_loss, val_loss]
    accuracies = [train_acc, val_acc]

    plt.style.use("ggplot")
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Bar Chart
    bars_loss = ax_loss.bar(categories, losses, color=['#1f77b4', '#d62728'], width=0.5)
    ax_loss.set_title("Loss Comparison", fontsize=13, fontweight='bold')
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars_loss:
        height = bar.get_height()
        ax_loss.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold')

    # Accuracy Bar Chart
    bars_acc = ax_acc.bar(categories, accuracies, color=['#1f77b4', '#2ca02c'], width=0.5)
    ax_acc.set_title("Accuracy Comparison (%)", fontsize=13, fontweight='bold')
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_ylim(0, 105)
    ax_acc.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars_acc:
        height = bar.get_height()
        ax_acc.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontweight='bold')

    diff_acc = train_acc - val_acc
    plt.suptitle(f"Saved Model Overfitting Check (Gap: {diff_acc:.2f}% accuracy)", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_path = "graphs/model_evaluation_chart.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[+] Saved evaluation chart to '{save_path}'")
    plt.close()


if __name__ == "__main__":
    main()