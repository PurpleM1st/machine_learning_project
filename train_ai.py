import numpy as np
import os
import joblib
import cv2
import gc
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed

from database import get_data

def extract_lbp_features(gray_img, P=8, R=1):
    """
    Computes Rotation-Invariant Uniform Local Binary Pattern (LBP) histogram.
    High performance for fine micro-textures (wrinkles, skin vs smooth fur).
    """
    lbp = local_binary_pattern(gray_img, P=P, R=R, method="uniform")
    # Uniform LBP with P=8 produces P + 2 = 10 distinct bins
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)

def extract_spatial_hsv(img_hsv, grid=(2, 2), bins=(16, 16)):
    """
    Extracts HSV histograms per spatial quadrant rather than globally.
    Prevents background colors from erasing animal color identities.
    """
    h, w, _ = img_hsv.shape
    dh, dw = h // grid[0], w // grid[1]
    spatial_hists = []

    for i in range(grid[0]):
        for j in range(grid[1]):
            cell = img_hsv[i*dh:(i+1)*dh, j*dw:(j+1)*dw]
            hist = cv2.calcHist([cell], [0, 1], None, bins, [0, 180, 0, 256])
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            spatial_hists.append(hist.flatten())

    return np.concatenate(spatial_hists).astype(np.float32)

def process_single_image(img):
    """
    3-Way Feature Fusion: Dense HOG (Shape) + LBP (Texture) + Spatial HSV (Color)
    """
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 1. Dense HOG (8x8 cells + 12 orientations for sharp edge detail)
    hog_feat = hog(
        gray, 
        orientations=12, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(2, 2), 
        block_norm='L2-Hys', 
        visualize=False
    )

    # 2. Texture Feature (LBP across 2 radii to capture skin micro-wrinkles)
    lbp_r1 = extract_lbp_features(gray, P=8, R=1)
    lbp_r2 = extract_lbp_features(gray, P=16, R=2)
    lbp_feat = np.hstack([lbp_r1, lbp_r2])

    # 3. Spatial Color Feature (2x2 Grid HSV Histogram)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    spatial_hsv_feat = extract_spatial_hsv(hsv, grid=(2, 2), bins=(8, 8))

    # Fusion into a single vector
    return np.hstack([hog_feat, lbp_feat, spatial_hsv_feat]).astype(np.float32)

def extract_features_parallel(image_array, batch_size=800):
    num_samples = len(image_array)
    print(f"Extracting Fused Features (HOG + LBP + Spatial HSV) for {num_samples} images...")

    features = Parallel(n_jobs=-1, batch_size=batch_size)(
        delayed(process_single_image)(img) for img in image_array
    )
    return np.array(features, dtype=np.float32)

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

    return np.array(labels, dtype=np.int64)

def train_and_save():
    print("=== Step 1: Loading Data & Generating Target Labels ===")
    train_x = get_data("train_x.npy")
    val_x = get_data("val_x.npy")

    y_train = derive_labels_fast("train")[:len(train_x)]
    y_val = derive_labels_fast("val")[:len(val_x)]

    print(f"Train X: {train_x.shape} | Train y: {y_train.shape}")
    print(f"Val X:   {val_x.shape} | Val y:   {y_val.shape}")

    print("\n=== Step 2: Parallel 3-Way Feature Extraction ===")
    X_train = extract_features_parallel(train_x)
    X_val = extract_features_parallel(val_x)

    del train_x, val_x
    gc.collect()

    print(f"Extracted feature dimension per image: {X_train.shape[1]}")

    print("\n=== Step 3: Feature Scaling ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    del X_train, X_val
    gc.collect()

    print("\n=== Step 4: Hyperparameter Tuning ===")
    candidate_c_values = [1.0, 5.0, 10.0]
    
    best_val_acc = -1.0
    best_svm_model = None
    best_c = None

    for c in candidate_c_values:
        print(f"\n--- Training RBF SVM with C = {c} ---")
        svm = SVC(
            kernel='rbf', 
            C=c, 
            gamma='scale', 
            cache_size=2000, 
            tol=1e-2,
            max_iter=8000
        )
        
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

    print(f"\n>>> Best Configuration: C = {best_c} (Validation Accuracy: {best_val_acc:.2f}%) <<<")

    print("\n=== Step 5: Saving Model Bundle ===")
    os.makedirs("models", exist_ok=True)
    model_bundle = {
        "scaler": scaler,
        "svm": best_svm_model,
        "best_c": best_c
    }
    joblib.dump(model_bundle, "models/trained_svm.pkl")
    print("Saved updated optimal model bundle to 'models/trained_svm.pkl'.")

if __name__ == "__main__":
    train_and_save()