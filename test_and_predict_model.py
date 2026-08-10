import os

import cv2
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from torchvision import models, transforms

from PIL import Image

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix
)


# Configuration
MODEL_PATH = "models/best_resnet18.pt"

TEST_DIR = "archive/animals/test"

BATCH_SIZE = 64
NUM_WORKERS = 4

OUTPUT_CONFUSION_MATRIX = (
	"graphs/confusion_matrix.png"
)


VALID_EXTS = {
	".jpg",
	".jpeg",
	".png",
	".bmp"
}


# 1. Dataset
class AnimalTestDataset(torch.utils.data.Dataset):
	def __init__(
		self,
		images_list,
		labels_list,
		transform=None
	):
		self.images = images_list
		self.labels = labels_list
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		img = self.images[idx]
		# Test images are read with cv2.imread(), so they are BGR.
		# Convert BGR -> RGB before PIL/torchvision.
		if isinstance(img, np.ndarray):
			if img.dtype != np.uint8:
				img = img.astype(
					np.uint8
				)
			img = cv2.cvtColor(
				img,
				cv2.COLOR_BGR2RGB
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


# 2. Load test dataset
def load_test_dataset(test_dir=TEST_DIR):
	"""
	Loads the test dataset.
	Test folders:
		cats_test       -> class 0 -> cat
		dogs_test       -> class 1 -> dog
		elephants_test  -> class 2 -> elephant
		horses_test     -> class 3 -> horse
		lions_test      -> class 4 -> lion
	"""

	if not os.path.isdir(test_dir):
		raise FileNotFoundError(
			f"Test directory not found:\n"
			f"  {test_dir}"
		)

	# Explicit mapping from test folder names to model class IDs
	TEST_FOLDER_TO_CLASS = {
		"cats_test": 0,
		"dogs_test": 1,
		"elephants_test": 2,
		"horses_test": 3,
		"lions_test": 4
	}

	# Find test folders
	folders = sorted(
		[
			f.name
			for f in os.scandir(test_dir)
			if f.is_dir()
		],
		key=str.lower
	)

	print("\n=== Test Folder Mapping ===")
	for folder in folders:
		if folder not in TEST_FOLDER_TO_CLASS:
			raise ValueError(
				f"\nUnknown test folder: {folder}\n\n"
				f"Expected folders:\n"
				f"  cats_test\n"
				f"  dogs_test\n"
				f"  elephants_test\n"
				f"  horses_test\n"
				f"  lions_test"
			)

		class_id = TEST_FOLDER_TO_CLASS[folder]

		print(
			f"  {folder} -> class {class_id}"
		)

	# Make sure no expected folder is missing
	missing_folders = [
		folder
		for folder in TEST_FOLDER_TO_CLASS
		if folder not in folders
	]

	if missing_folders:

		raise ValueError(
			"\nMissing test folders:\n"
			+ "\n".join(
				f"  {folder}"
				for folder in missing_folders
			)
		)

	# Load images
	images = []
	y_true = []

	print(
		f"\nScanning test set:\n"
		f"  {test_dir}"
	)

	# Process in class-ID order
	folders_by_class = sorted(
		folders,
		key=lambda folder:
			TEST_FOLDER_TO_CLASS[folder]
	)

	for folder in folders_by_class:
		folder_path = os.path.join(
			test_dir,
			folder
		)

		class_id = TEST_FOLDER_TO_CLASS[
			folder
		]

		# Deterministic filename ordering
		files = sorted(
			[
				f
				for f in os.scandir(folder_path)
				if (
					f.is_file()
					and os.path.splitext(
						f.name
					)[1].lower()
					in VALID_EXTS
				)
			],
			key=lambda f: f.name.lower()
		)

		file_count = 0

		for entry in files:
			img = cv2.imread(
				entry.path
			)

			if img is None:
				print(
					f"WARNING: Could not read:\n"
					f"  {entry.path}"
				)
				continue
			# Keep as BGR here.
			# AnimalTestDataset converts BGR -> RGB
			# before passing the image to PIL.

			images.append(img)
			y_true.append(class_id)
			file_count += 1
		print(f"  Class {class_id}: {folder}: {file_count} images")

	y_true = np.array(
		y_true,
		dtype=np.int64
	)

	# Sanity check
	if len(images) != len(y_true):
		raise ValueError(
			"\nImage/label count mismatch!\n"
			f"Images: {len(images)}\n"
			f"Labels: {len(y_true)}"
		)
	print(
		f"\nTotal test images: "
		f"{len(images)}"
	)

	# Return mapping using TRAINING class names
	test_class_map = {
		"cat": 0,
		"dog": 1,
		"elephant": 2,
		"horse": 3,
		"lion": 4
	}

	return (
		images,
		y_true,
		test_class_map
	)


# 3. Build model
def build_model(num_classes):
	model = models.resnet18(
		weights=None
	)
	in_features = (
		model.fc.in_features
	)
	model.fc = nn.Sequential(
		nn.Dropout(0.3),
		nn.Linear(
			in_features,
			num_classes
		)
	)
	return model


# 4. Plot confusion matrices
def plot_confusion_matrices(y_true, y_pred, class_names, output_path=OUTPUT_CONFUSION_MATRIX):
	# Raw confusion matrix
	cm_raw = confusion_matrix(
		y_true,
		y_pred
	)

	# Row-normalized confusion matrix
	cm_norm = confusion_matrix(
		y_true,
		y_pred,
		normalize="true"
	)

	# Create figure
	fig, axes = plt.subplots(
		1,
		2,
		figsize=(16, 6)
	)

	# Raw counts
	sns.heatmap(
		cm_raw,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=class_names,
		yticklabels=class_names,
		ax=axes[0]
	)

	axes[0].set_title(
		"Confusion Matrix (Counts)"
	)

	axes[0].set_xlabel(
		"Predicted Label"
	)

	axes[0].set_ylabel(
		"True Label"
	)

	# Normalized percentages
	sns.heatmap(
		cm_norm,
		annot=True,
		fmt=".2%",
		cmap="Greens",
		xticklabels=class_names,
		yticklabels=class_names,
		ax=axes[1]
	)

	axes[1].set_title(
		"Confusion Matrix (Normalized Ratio)"
	)
	axes[1].set_xlabel(
		"Predicted Label"
	)
	axes[1].set_ylabel(
		"True Label"
	)

	# Save
	plt.tight_layout()
	os.makedirs(
		os.path.dirname(output_path),
		exist_ok=True
	)
	plt.savefig(
		output_path,
		dpi=300,
		bbox_inches="tight"
	)
	plt.close()
	print(
		f"\nSaved confusion matrix to:"
		f"\n  {output_path}"
	)


# 5. Evaluate test set
def evaluate_test_set(model_path=MODEL_PATH):
	# Device
	device = torch.device(
		"cuda"
		if torch.cuda.is_available()
		else "cpu"
	)

	os.makedirs("graphs", exist_ok=True)

	print(
		f"Using device: {device}"
	)

	# STEP 1: Load checkpoint
	print(
		"\n=== Step 1: Loading Model ==="
	)

	if not os.path.exists(model_path):
		raise FileNotFoundError(
			f"Model file not found:\n"
			f"  {model_path}\n\n"
			"Train the model first."
		)

	checkpoint = torch.load(
		model_path,
		map_location=device
	)

	num_classes = checkpoint["num_classes"]

	print(
		f"Number of classes: "
		f"{num_classes}"
	)

	# Load class mapping saved during training
	if "class_map" in checkpoint:
		saved_class_map = (
			checkpoint["class_map"]
		)
		print(
			"\nClass mapping stored "
			"in checkpoint:"
		)

		for name, class_id in (
			saved_class_map.items()
		):

			print(f"  {class_id} -> {name}")
	else:
		saved_class_map = {
			"cat": 0,
			"dog": 1,
			"elephant": 2,
			"horse": 3,
			"lion": 4
		}

		print("\nWARNING:")

		print(
			"Checkpoint does not contain "
			"class mapping."
		)

	# STEP 2: Build model
	model = build_model(
		num_classes=num_classes
	)

	model.load_state_dict(
		checkpoint[
			"model_state_dict"
		]
	)

	model = model.to(
		device
	)

	model.eval()

	print(
		"\nModel loaded successfully."
	)

	if "val_acc" in checkpoint:
		print(
			f"Saved validation accuracy: "
			f"{checkpoint['val_acc']:.2f}%"
		)

	# STEP 3: Load test images
	print(
		"\n=== Step 2: Loading Test Images ==="
	)

	(
		images,
		y_true,
		test_class_map
	) = load_test_dataset()

	if len(images) == 0:

		raise ValueError(
			"No test images were loaded."
		)

	# Check class mapping consistency
	checkpoint_class_map = {
		name: int(class_id)
		for name, class_id
		in saved_class_map.items()
	}

	if checkpoint_class_map != test_class_map:
		print(
			"\nWARNING: CLASS MAPPING MISMATCH!"
		)
		print(
			f"Training mapping:"
			f"\n  {checkpoint_class_map}"
		)
		print(
			f"Test mapping:"
			f"\n  {test_class_map}"
		)
		raise ValueError(
			"\nThe class mapping used during "
			"training is different from the "
			"test class mapping."
		)

	print(
		"\nTraining and test class mappings match."
	)

	# STEP 4: Test transforms
	test_transforms = transforms.Compose([
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

	# STEP 5: Dataset and DataLoader
	test_dataset = AnimalTestDataset(
		images,
		y_true,
		transform=test_transforms
	)

	test_loader = DataLoader(
		test_dataset,
		batch_size=BATCH_SIZE,
		shuffle=False,
		num_workers=NUM_WORKERS,
		pin_memory=torch.cuda.is_available()
	)

	# STEP 6: Generate predictions
	print(
		f"\n=== Step 3: Generating "
		f"Predictions ==="
	)

	y_pred = []

	with torch.no_grad():
		for batch_images, _ in test_loader:
			batch_images = batch_images.to(
				device,
				non_blocking=True
			)
			outputs = model(
				batch_images
			)
			_, predicted = outputs.max(
				1
			)
			y_pred.extend(
				predicted.cpu().numpy()
			)

	y_pred = np.array(
		y_pred,
		dtype=np.int64
	)

	# STEP 7: Calculate accuracy
	test_acc = (
		accuracy_score(
			y_true,
			y_pred
		)
		* 100
	)

	# Create class names in index order
	class_names = [
		name
		for name, class_id
		in sorted(
			test_class_map.items(),
			key=lambda item: item[1]
		)
	]

	# STEP 8: Print results
	print("\n" + "=" * 60)
	print(f"OVERALL TEST ACCURACY: {test_acc:.2f}%")
	print("=" * 60)

	# Classification report
	print(
		"\nDetailed Classification Report:"
	)

	print(
		classification_report(
			y_true,
			y_pred,
			labels=list(
				range(num_classes)
			),
			target_names=class_names,
			digits=4,
			zero_division=0
		)
	)

	# STEP 9: Confusion matrix
	print(
		"=== Step 4: Generating "
		"Confusion Matrices ==="
	)

	os.makedirs(
		"models",
		exist_ok=True
	)

	plot_confusion_matrices(
		y_true,
		y_pred,
		class_names
	)

	# STEP 10: Print per-class accuracy
	print(
		"\n=== Per-Class Accuracy ==="
	)

	cm = confusion_matrix(
		y_true,
		y_pred,
		labels=list(
			range(num_classes)
		)
	)

	for class_id, class_name in enumerate(class_names):
		total = cm[class_id].sum()
		correct = cm[
			class_id,
			class_id
		]
		if total > 0:
			accuracy = (
				correct
				/ total
				* 100
			)
		else:
			accuracy = 0.0

		print(
			f"  {class_id} -> "
			f"{class_name}: "
			f"{accuracy:.2f}% "
			f"({correct}/{total})"
		)

	print(
		"\nEvaluation complete."
	)


# Main
if __name__ == "__main__":
	evaluate_test_set()