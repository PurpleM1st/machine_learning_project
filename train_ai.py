import os
import gc
<<<<<<< HEAD
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed

from database import get_data


def process_single_image(img):
    """
    Extracts HOG shape features + HSV color histogram for a single image.
    """
    img_resized = cv2.resize(img, (128, 128))

    # 1. HOG features (Grayscale shape/edges)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
=======
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
>>>>>>> refs/remotes/origin/Alex_environment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        # Ensure image is uint8 PIL Image for torchvision transforms
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = Image.fromarray(img)

<<<<<<< HEAD

def extract_features_parallel(image_array, batch_size=1000):
    """
    Extracts features in parallel across all CPU cores for maximum speed.
    """
    num_samples = len(image_array)
    print(f"Extracting features for {num_samples} images in parallel...")

    features = Parallel(n_jobs=-1, batch_size=batch_size)(
        delayed(process_single_image)(img) for img in image_array
    )

    return np.array(features, dtype=np.float32)


def derive_labels_fast(split_name):
    """
    Fast label derivation using directory scan instead of slow image opens.

    Returns both the label array and the class_map (folder name -> class id)
    used to build it, so the mapping can be persisted alongside the model.
    """
=======
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label

# ---------------------------------------------------------------------
# 2. Fast Label Derivation
# ---------------------------------------------------------------------
def derive_labels_fast(split_name):
>>>>>>> refs/remotes/origin/Alex_environment
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

<<<<<<< HEAD
    return np.array(labels, dtype=np.int64), class_map


def derive_labels_checked(split_name, image_array):
    """
    Wraps derive_labels_fast with a hard check that the directory-scan label
    order actually lines up with the pre-extracted .npy array. Silently
    slicing labels to match image_array length (the old `[:len(train_x)]`
    trick) hides a length mismatch instead of catching it, and a mismatch
    here means every label could be silently wrong. If the counts don't
    match exactly, we fail loudly rather than guess.
    """
    labels, class_map = derive_labels_fast(split_name)

    if len(labels) != len(image_array):
        raise ValueError(
            f"[{split_name}] Label/image count mismatch: "
            f"found {len(labels)} labels via directory scan but "
            f"{len(image_array)} images in the .npy array. "
            f"This almost certainly means the .npy file's image order "
            f"does not correspond 1:1 with the os.scandir() folder order "
            f"used here. Do not silently truncate labels to fit — "
            f"regenerate train_x.npy/val_x.npy with a pipeline that "
            f"records labels at extraction time, or otherwise confirm "
            f"the ordering explicitly before training."
        )

    return labels, class_map

=======
    return np.array(labels, dtype=np.int64), len(folders)
>>>>>>> refs/remotes/origin/Alex_environment

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

<<<<<<< HEAD
    print("Generating 1D label vectors (y)...")
    y_train, class_map_train = derive_labels_checked("train", train_x)
    y_val, class_map_val = derive_labels_checked("val", val_x)

    if class_map_train != class_map_val:
        raise ValueError(
            "Train and val class_maps disagree — the folder names/order "
            "under archive/animals/train and archive/animals/val don't "
            "match, so class indices would not be comparable between "
            "splits."
        )
    class_map = class_map_train

    print(f"y_train shape: {y_train.shape} | y_val shape: {y_val.shape}")
    print(f"Class map: {class_map}")
=======
    y_train = y_train[:len(train_x)]
    y_val = y_val[:len(val_x)]

    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)} | Classes: {num_classes}")
>>>>>>> refs/remotes/origin/Alex_environment

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

<<<<<<< HEAD
    print("\n=== Step 4: Training RBF SVM (grid search C x gamma) ===")

    # Wider, more meaningful grid over both C and gamma.
    candidate_c_values = [1, 10, 100]
    candidate_gamma_values = ["scale", 0.001, 0.01, 0.1]

    # Set True if your classes are meaningfully imbalanced (see class counts
    # printed above / in the classification_report below).
    use_class_weight_balanced = False

    best_val_acc = -1.0
    best_model = None
    best_c = None
    best_gamma = None
    best_val_preds = None

    for c in candidate_c_values:
        for gamma in candidate_gamma_values:
            print(f"\n--- Training RBF SVM with C={c}, gamma={gamma} ---")

            model = SVC(
                kernel='rbf',
                C=c,
                gamma=gamma,
                cache_size=2000,
                tol=1e-2,
                max_iter=6000,
                class_weight="balanced" if use_class_weight_balanced else None,
            )
            model.fit(X_train_scaled, y_train)

            val_preds = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_preds) * 100

            print(f"Results for C={c:<5} gamma={str(gamma):<6} -> Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_c = c
                best_gamma = gamma
                best_val_preds = val_preds
            else:
                del model

            gc.collect()

    print(f"\n>>> Best Configuration Selected: C={best_c}, gamma={best_gamma} "
          f"(Validation Accuracy: {best_val_acc:.2f}%) <<<")

    print("\n=== Step 4b: Per-Class Report (checks for class-imbalance issues) ===")
    target_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    print(classification_report(y_val, best_val_preds, target_names=target_names))

    # Note: probability estimates were dropped along with CalibratedClassifierCV
    # for runtime cost. If you need predict_proba later, either pass
    # probability=True to SVC (adds its own overhead via internal 5-fold CV
    # at fit time) or reintroduce calibration selectively on just the final
    # chosen model rather than inside the grid search.

    print("\n=== Step 5: Saving Model Bundle ===")
    os.makedirs("models", exist_ok=True)
    model_bundle = {
        "scaler": scaler,
        "svm": best_model,  # tuned SVC (uncalibrated — see note above)
        "best_c": best_c,
        "best_gamma": best_gamma,
        "image_size": (128, 128),
        "hog_orientations": 9,
        "hog_pixels_per_cell": (16, 16),
        "hog_cells_per_block": (2, 2),
        "hsv_bins": (16, 16),
        "class_map": class_map,
    }
    joblib.dump(model_bundle, "models/trained_svm.pkl")
    print("Saved optimal model bundle to 'models/trained_svm.pkl'.")
=======
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
>>>>>>> refs/remotes/origin/Alex_environment


if __name__ == "__main__":
    train_and_save()