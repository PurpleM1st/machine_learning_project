import numpy as np
import os
import joblib
import cv2
import gc
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

    # 2. HSV Color Histogram
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hsv_feat = hist.flatten()

    return np.hstack([hog_feat, hsv_feat]).astype(np.float32)


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


def train_and_save():
    print("=== Step 1: Loading Train and Validation Splits ===")
    train_x = get_data("train_x.npy")
    val_x = get_data("val_x.npy")

    print(f"Loaded Train X: {train_x.shape}")
    print(f"Loaded Val X:   {val_x.shape}")

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


if __name__ == "__main__":
    train_and_save()