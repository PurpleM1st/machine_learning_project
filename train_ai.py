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

# This is used so that the image data that we have and want to train
# the AI on is made in a way meant to be easier to comprehend for PyTorch
# while labelling them properly based on the name
class AnimalDataset(Dataset):
    def __init__(self, images_array, labels_array, transform=None):
        self.images = images_array
        self.labels = labels_array
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # syntax to be used later as it requires the proper name
        img = self.images[idx]
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            img = Image.fromarray(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label

# This function is used to take out the labels from the images name so that 
# the Ai can properly convert them to simple numbers while connecting them to
# the desired images. It check through folders, subfolders before eventually
# reaching the location of the images
def derive_labels_fast(split_name):
    path = f"archive/animals/{split_name}"
    labels = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    folders = sorted([f.name for f in os.scandir(path) if f.is_dir()])

	# creates the enumerators connecting the images to the numbers after the sorting
    class_map = {folder_name: idx for idx, folder_name in enumerate(folders)}

    for folder in folders:
        sub_path = os.path.join(path, folder)
        class_id = class_map[folder] # getting the numerical class
        for entry in os.scandir(sub_path):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts:
                labels.append(class_id) # splitting the names and appending them to be used by the AI

    return np.array(labels, dtype=np.int64), len(folders)

# This is the core function that trains the AI on the given dataset
# We use pretrained=True because ResNet18 is already used to images, but
# it cna be easily changed for trying the not trained model, although it has better
# results if the model is already computed from the library
def build_model(num_classes, pretrained=True):
    # Loads the already existing ResNet18 model
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3), # randomly substitutes 0 so it reduces overfitting so it does not become dependent
        nn.Linear(in_features, num_classes) # takes the features and the number of possible outcomes
    )
    return model

# This is the main pipeline that calls the necessary previous functions
# It is used to get the images, train the model while also applying custom
# implementations with the goal of improvement.
def train_and_save():

	# this is optional as its only job is to decide between the cpu and cuba for the
    # hardware necessary for the training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== Step 1: Loading Data & Deriving Labels ===")
    train_x = get_data("train_x.npy")
    val_x = get_data("val_x.npy")

    y_train, num_classes = derive_labels_fast("train") # gets the labels from the train dataset
    y_val, _ = derive_labels_fast("val") # gets the label for the validation nd temporary prediction

    y_train = y_train[:len(train_x)]
    y_val = y_val[:len(val_x)]

    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)} | Classes: {num_classes}")

    # Data Transformation, randomizing the rotation as well as customizing with constants that
    # have been tested to work the best, getting multiple rotations so the model
    # can analyze multiple outcomes even though in reality the animals can't stand on their head
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

	# this doesn't transfrom through rotations as we want to see how it would fare in reality
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
    criterion = nn.CrossEntropyLoss()  # https://www.pinecone.io/learn/cross-entropy-loss/
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2) # better than the normal gradient descent
    # as it uses all the previous weights to calculate the best possible movement. Learning rate has been chosen through
    # trial and error
    

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10) # used for the changing of the learning
    # rate through the cosine functions, making it smaller and smaller throughout the process as we get closer and closer to
    # the desired answer and minimization of the pre constructed errors that appear

    epochs = 10 # equal to T_max for a clean optimization
    best_val_acc = 0.0

    print("\n=== Step 3: Training Loop ===")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0 # for the precision

		# We begin training it on the train set and calculate the predictions. By having some reduced data,
        # parameters
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device) # getting the images

            optimizer.zero_grad()            # Clear gradients as we only want the current gradient here, otherwise they accumulate
            outputs = model(images)          # Forward pass the images in batch sizes
            loss = criterion(outputs, labels) # Uses the Entropy function for calculating the error on the current step
            loss.backward()                  # Backward pass, calculating the gradients that will be later used
            optimizer.step()                 # This uses the calculated gradients and pplies the AdamW algorithm or modifying the weights

            running_loss += loss.item() * images.size(0) # calculates the overall loss in order for clear precision
            # it uses the average (loss.item()) and multiplies by the number of images in a batch as otherwise it would
			# take decades
            _, predicted = outputs.max(1) # gets the indices of the most prominent class, thus the predicted one
            total += labels.size(0) # total of images loaded and trained
            correct += predicted.eq(labels).sum().item() # calcultes the number of correct labelization and adds it to the total

        scheduler.step() # updates the learning rate that we have scheduled
        train_acc = (correct / total) * 100 # percentage of correct labels
        train_loss = running_loss / total # the training loss, how many predicions are wrong

        # Now we move to the validation test in order to see how the model reacts to it
        # Thus, we stop training it and begin evaluating it
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad(): # stopping the gradient calculations
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0) # same as before, we calculate the
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = (val_correct / val_total) * 100 # accuracy for the validation
        epoch_val_loss = val_loss / val_total

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # We save the best model to the device
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