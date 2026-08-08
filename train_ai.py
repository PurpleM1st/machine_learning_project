import os
import gc

import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from PIL import Image

import numpy as np

from database import get_data


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TRAIN_DIR = "archive/animals/train"
VAL_DIR = "archive/animals/val"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_resnet18.pt"
)

BATCH_SIZE = 64
NUM_WORKERS = 4

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2

EPOCHS = 10


# ---------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------

class AnimalDataset(Dataset):

    def __init__(
        self,
        images_array,
        labels_array,
        transform=None
    ):

        self.images = images_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        img = self.images[idx]

        # -------------------------------------------------------------
        # Images coming from database.py are ALREADY RGB.
        #
        # DO NOT use cv2.cvtColor() here.
        # -------------------------------------------------------------

        if isinstance(img, np.ndarray):

            if img.dtype != np.uint8:

                img = img.astype(
                    np.uint8
                )

            img = Image.fromarray(
                img
            )

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        if self.transform:

            img = self.transform(
                img
            )

        return img, label


# ---------------------------------------------------------------------
# 2. Deterministic label generation
# ---------------------------------------------------------------------

def derive_labels_fast(split_name):

    path = os.path.join(
        "archive",
        "animals",
        split_name
    )

    if not os.path.isdir(path):

        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"  {path}"
        )

    valid_exts = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # This MUST use the exact same folder ordering as database.py.
    # -------------------------------------------------------------

    folders = sorted(
        [
            f.name
            for f in os.scandir(path)
            if f.is_dir()
        ],
        key=str.lower
    )

    class_map = {
        folder_name: idx
        for idx, folder_name in enumerate(folders)
    }

    labels = []

    print(
        f"\nClass mapping for {split_name}:"
    )

    for folder_name, class_id in class_map.items():

        print(
            f"  {class_id} -> {folder_name}"
        )

        folder_path = os.path.join(
            path,
            folder_name
        )

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Same deterministic file ordering as database.py.
        # ---------------------------------------------------------

        files = sorted(
            [
                f
                for f in os.scandir(folder_path)
                if (
                    f.is_file()
                    and os.path.splitext(
                        f.name
                    )[1].lower()
                    in valid_exts
                )
            ],
            key=lambda f: f.name.lower()
        )

        labels.extend(
            [class_id] * len(files)
        )

    labels = np.array(
        labels,
        dtype=np.int64
    )

    return labels, len(folders), class_map


# ---------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------

def build_model(
    num_classes,
    pretrained=True
):

    if pretrained:

        weights = (
            models.ResNet18_Weights.DEFAULT
        )

    else:

        weights = None

    model = models.resnet18(
        weights=weights
    )

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            in_features,
            num_classes
        )
    )

    return model


# ---------------------------------------------------------------------
# 4. Main training pipeline
# ---------------------------------------------------------------------

def train_and_save():

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    print(
        "\n=== Step 1: Loading Data ==="
    )

    train_x = get_data(
        "train_x.npy"
    )

    val_x = get_data(
        "val_x.npy"
    )

    # -------------------------------------------------------------
    # Generate labels
    # -------------------------------------------------------------

    y_train, num_classes, train_class_map = (
        derive_labels_fast("train")
    )

    y_val, val_num_classes, val_class_map = (
        derive_labels_fast("val")
    )

    # -------------------------------------------------------------
    # Verify class mappings
    # -------------------------------------------------------------

    print(
        "\n=== Checking Class Mappings ==="
    )

    print(
        f"Train mapping: {train_class_map}"
    )

    print(
        f"Val mapping:   {val_class_map}"
    )

    if train_class_map != val_class_map:

        raise ValueError(
            "\nTRAIN AND VALIDATION CLASS MAPPINGS "
            "ARE DIFFERENT!\n\n"
            f"Train: {train_class_map}\n"
            f"Val:   {val_class_map}"
        )

    if num_classes != val_num_classes:

        raise ValueError(
            "Train and validation have different "
            "numbers of classes."
        )

    # -------------------------------------------------------------
    # Verify image/label counts
    # -------------------------------------------------------------

    print(
        "\n=== Dataset Sanity Check ==="
    )

    print(
        f"Train images: {len(train_x)}"
    )

    print(
        f"Train labels: {len(y_train)}"
    )

    print(
        f"Val images:   {len(val_x)}"
    )

    print(
        f"Val labels:   {len(y_val)}"
    )

    if len(train_x) != len(y_train):

        raise ValueError(
            "\nTRAIN IMAGE/LABEL COUNT MISMATCH!\n"
            f"Images: {len(train_x)}\n"
            f"Labels: {len(y_train)}\n\n"
            "Do not continue until this is fixed."
        )

    if len(val_x) != len(y_val):

        raise ValueError(
            "\nVALIDATION IMAGE/LABEL COUNT MISMATCH!\n"
            f"Images: {len(val_x)}\n"
            f"Labels: {len(y_val)}\n\n"
            "Do not continue until this is fixed."
        )

    print(
        "\nImage and label counts match."
    )

    # -------------------------------------------------------------
    # Print class distribution
    # -------------------------------------------------------------

    print(
        "\n=== Training Class Distribution ==="
    )

    for class_name, class_id in train_class_map.items():

        count = int(
            np.sum(
                y_train == class_id
            )
        )

        print(
            f"  {class_id} -> "
            f"{class_name}: "
            f"{count} images"
        )

    # -------------------------------------------------------------
    # Verify NumPy image format
    # -------------------------------------------------------------

    print(
        "\n=== Image Format ==="
    )

    print(
        f"Train shape: {train_x.shape}"
    )

    print(
        f"Train dtype: {train_x.dtype}"
    )

    print(
        f"Val shape:   {val_x.shape}"
    )

    print(
        f"Val dtype:   {val_x.dtype}"
    )

    if train_x.ndim != 4:

        raise ValueError(
            f"Unexpected training shape: "
            f"{train_x.shape}"
        )

    if train_x.shape[-1] != 3:

        raise ValueError(
            "Training images do not have "
            "3 color channels."
        )

    # -------------------------------------------------------------
    # Data transforms
    # -------------------------------------------------------------

    print(
        "\n=== Creating Data Transforms ==="
    )

    train_transforms = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(
            15
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    val_transforms = transforms.Compose([

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        )
    ])

    # -------------------------------------------------------------
    # Create datasets
    # -------------------------------------------------------------

    train_dataset = AnimalDataset(
        train_x,
        y_train,
        transform=train_transforms
    )

    val_dataset = AnimalDataset(
        val_x,
        y_val,
        transform=val_transforms
    )

    # -------------------------------------------------------------
    # Data loaders
    # -------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    print(
        "\n=== Step 2: Initializing ResNet-18 ==="
    )

    model = build_model(
        num_classes=num_classes,
        pretrained=True
    )

    model = model.to(device)

    # -------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # -------------------------------------------------------------
    # Learning-rate scheduler
    # -------------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # -------------------------------------------------------------
    # Training settings
    # -------------------------------------------------------------

    best_val_acc = 0.0

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    print(
        "\n=== Step 3: Training ==="
    )

    # =============================================================
    # Training loop
    # =============================================================

    for epoch in range(EPOCHS):

        # ---------------------------------------------------------
        # TRAIN
        # ---------------------------------------------------------

        model.train()

        running_loss = 0.0

        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            # Clear gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(
                images
            )

            # Loss
            loss = criterion(
                outputs,
                labels
            )

            # Backward
            loss.backward()

            # Gradient descent
            optimizer.step()

            # -----------------------------------------------------
            # Statistics
            # -----------------------------------------------------

            batch_size = images.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            _, predicted = outputs.max(
                1
            )

            train_total += labels.size(0)

            train_correct += (
                predicted
                .eq(labels)
                .sum()
                .item()
            )

        scheduler.step()

        train_loss = (
            running_loss
            / train_total
        )

        train_acc = (
            train_correct
            / train_total
            * 100
        )

        # ---------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------

        model.eval()

        val_running_loss = 0.0

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                labels = labels.to(
                    device,
                    non_blocking=True
                )

                outputs = model(
                    images
                )

                loss = criterion(
                    outputs,
                    labels
                )

                batch_size = images.size(0)

                val_running_loss += (
                    loss.item()
                    * batch_size
                )

                _, predicted = outputs.max(
                    1
                )

                val_total += labels.size(0)

                val_correct += (
                    predicted
                    .eq(labels)
                    .sum()
                    .item()
                )

        val_loss = (
            val_running_loss
            / val_total
        )

        val_acc = (
            val_correct
            / val_total
            * 100
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ---------------------------------------------------------
        # Print results
        # ---------------------------------------------------------

        print(
            f"\nEpoch [{epoch + 1:02d}/{EPOCHS:02d}]"
        )

        print(
            f"  Learning Rate: {current_lr:.8f}"
        )

        print(
            f"  Train Loss:    {train_loss:.4f}"
        )

        print(
            f"  Train Acc:     {train_acc:.2f}%"
        )

        print(
            f"  Val Loss:      {val_loss:.4f}"
        )

        print(
            f"  Val Acc:       {val_acc:.2f}%"
        )

        # ---------------------------------------------------------
        # Save best model
        # ---------------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            checkpoint = {

                "model_state_dict":
                    model.state_dict(),

                "val_acc":
                    val_acc,

                "num_classes":
                    num_classes,

                "class_map":
                    train_class_map,

                "input_size":
                    224,

                "mean":
                    [
                        0.485,
                        0.456,
                        0.406
                    ],

                "std":
                    [
                        0.229,
                        0.224,
                        0.225
                    ]
            }

            torch.save(
                checkpoint,
                MODEL_PATH
            )

            print(
                f"  *** New best model saved "
                f"({val_acc:.2f}%) ***"
            )

    # -------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        f"BEST VALIDATION ACCURACY: "
        f"{best_val_acc:.2f}%"
    )

    print(
        "=" * 60
    )

    print(
        f"Model saved to:\n"
        f"  {MODEL_PATH}"
    )

    # -------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------

    del train_x
    del val_x

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    train_and_save()