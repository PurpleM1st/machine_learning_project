import numpy as np
import os
import joblib
import cv2
import gc
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

    # 2. HSV Color Histogram
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hsv_feat = hist.flatten()

    return np.hstack([hog_feat, hsv_feat]).astype(np.float32)

def extract_features_parallel(image_array, batch_size=400):
    """
    Extracts features in parallel across all CPU cores for maximum speed.
    """
    num_samples = len(image_array)
    print(f"Extracting features for {num_samples} images in parallel...")

    features = Parallel(n_jobs=2, batch_size=batch_size)(
        delayed(process_single_image)(img) for img in image_array
    )
    
    return np.array(features, dtype=np.float32)

def derive_labels_fast(split_name):
    """
    Fast label derivation using directory scan instead of slow image opens.
    """
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

    return np.array(labels, dtype=np.int64)

def train_and_save():
    print("=== Step 1: Loading Train and Validation Splits ===")
    train_x = get_data("train_x.npy")
    val_x = get_data("val_x.npy")

    print(f"Loaded Train X: {train_x.shape}")
    print(f"Loaded Val X:   {val_x.shape}")

    print("Generating 1D label vectors (y)...")
    y_train = derive_labels_fast("train")[:len(train_x)]
    y_val = derive_labels_fast("val")[:len(val_x)]

    print(f"y_train shape: {y_train.shape} | y_val shape: {y_val.shape}")

    print("\n=== Step 2: Extracting Features (Parallel HOG + HSV) ===")
    X_train = extract_features_parallel(train_x)
    X_val = extract_features_parallel(val_x)

    # Free raw pixel RAM
    del train_x, val_x
    gc.collect()

    print("\n=== Step 3: Preprocessing (Fitting Scaler ONLY on Train) ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    del X_train, X_val
    gc.collect()

    print("\n=== Step 4: Training RBF SVM ===")
    # Dropped 0.1 to avoid slow non-convergence loops
    candidate_c_values = [1.0, 5.0, 10.0]
    
    best_val_acc = -1.0
    best_svm_model = None
    best_c = None

    for c in candidate_c_values:
        print(f"\n--- Training RBF SVM with C = {c} ---")
        
        # max_iter=10000 ensures it will NEVER hang indefinitely
        svm = SVC(
            kernel='rbf', 
            C=c, 
            gamma='scale', 
            cache_size=2000, 
            tol=1e-2,
            max_iter=6000
        )
        CalibratedClassifierCV(SVC(), ensemble=False)
        
        svm.fit(X_train_scaled, y_train)
        
        val_preds = svm.predict(X_val_scaled)
        val_acc = accuracy_score(y_val, val_preds) * 100
        
        print(f"Results for C={c:<4} -> Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_svm_model = svm
            best_c = c
        else:
            del svm
            
        gc.collect()

    print(f"\n>>> Best Configuration Selected: C = {best_c} (Validation Accuracy: {best_val_acc:.2f}%) <<<")

    print("\n=== Step 5: Saving Model Bundle ===")
    os.makedirs("models", exist_ok=True)
    model_bundle = {
        "scaler": scaler,
        "svm": best_svm_model,
        "best_c": best_c
    }
    joblib.dump(model_bundle, "models/trained_svm.pkl")
    print("Saved optimal model bundle to 'models/trained_svm.pkl'.")

if __name__ == "__main__":
    train_and_save()