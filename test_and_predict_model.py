import os
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import hog
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import Parallel, delayed

# Map class index (0-4) directly to animal names
CLASS_MAPPING = {
    0: "cat",
    1: "dog",
    2: "elephant",
    3: "horse",
    4: "lion"
}

def process_single_image(img):
    """
    Extracts HOG shape features + HSV color histogram for a single image.
    """
    img_resized = cv2.resize(img, (128, 128))
    
    # 1. HOG features
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(
        gray, 
        orientations=9, 
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        visualize=False
    )

    # 2. HSV Color Histogram
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hsv_feat = hist.flatten()

    return np.hstack([hog_feat, hsv_feat]).astype(np.float32)

def load_test_dataset(test_dir="archive/animals/test"):
    """
    Loads all images and ground-truth labels directly from the test subfolders.
    """
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Sort folders to ensure consistent indexing matching CLASS_MAPPING
    folders = sorted([f.name for f in os.scandir(test_dir) if f.is_dir()])
    
    images = []
    y_true = []
    
    print(f"Scanning test set directory: '{test_dir}'...")
    
    for class_id, folder in enumerate(folders):
        folder_path = os.path.join(test_dir, folder)
        file_count = 0
        
        for entry in os.scandir(folder_path):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_exts:
                img = cv2.imread(entry.path)
                if img is not None:
                    images.append(img)
                    y_true.append(class_id)
                    file_count += 1
                    
        print(f"  Found {file_count:>4} images for class {class_id} ({CLASS_MAPPING.get(class_id, folder)})")

    return images, np.array(y_true, dtype=np.int64)

def plot_confusion_matrices(y_true, y_pred, class_names, output_path="models/confusion_matrix.png"):
    """
    Generates and saves side-by-side raw and normalized confusion matrix plots.
    """
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts plot
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Normalized percentages plot
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalized Ratio)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrix heatmaps to '{output_path}'")

def evaluate_test_set(model_path="models/trained_svm.pkl"):
    # 1. Load trained pipeline bundle
    print("=== Step 1: Loading Trained Model Bundle ===")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first!")

    bundle = joblib.load(model_path)
    scaler = bundle["scaler"]
    svm_model = bundle["svm"]

    # 2. Load test set images & labels
    print("\n=== Step 2: Loading Test Images ===")
    images, y_true = load_test_dataset()
    
    if len(images) == 0:
        raise ValueError("No test images were loaded. Check path 'archive/animals/test'")

    # 3. Parallel Feature Extraction
    print(f"\n=== Step 3: Extracting Features for {len(images)} Test Images ===")
    features = Parallel(n_jobs=-1, batch_size=500)(
        delayed(process_single_image)(img) for img in images
    )
    X_test = np.array(features, dtype=np.float32)

    # 4. Scale and Predict
    print("\n=== Step 4: Generating Predictions ===")
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test_scaled)

    # 5. Calculate Metrics
    test_acc = accuracy_score(y_true, y_pred) * 100
    class_names = [CLASS_MAPPING[i] for i in sorted(CLASS_MAPPING.keys())]

    print("\n" + "=" * 55)
    print(f"       OVERALL TEST ACCURACY: {test_acc:.2f}%")
    print("=" * 55)

    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=2))

    # 6. Plot and save confusion matrices
    print("=== Step 5: Generating Confusion Matrix Visualizations ===")
    plot_confusion_matrices(y_true, y_pred, class_names)

if __name__ == "__main__":
    evaluate_test_set()