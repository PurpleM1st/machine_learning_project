import os
import cv2
import numpy as np


# Configuration
CACHE_DIR = "cache"

VALID_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}

# Change this whenever the way images are cached changes.
CACHE_VERSION = "RGB_V2"

# Helper: Get image files in deterministic order
def get_image_files(folder_path):
    """
    Returns image files in deterministic alphabetical order.
    """

    files = [
        f
        for f in os.scandir(folder_path)
        if (
            f.is_file()
            and os.path.splitext(f.name)[1].lower() in VALID_EXTS
        )
    ]

    return sorted(
        files,
        key=lambda f: f.name.lower()
    )


# Load all images from a dataset split
def load_path(path):
    """
    Loads all images from:
        archive/animals/<split>

    Images are:
        1. Read using OpenCV
        2. Converted BGR -> RGB
        3. Resized to 224x224
        4. Stored as uint8 NumPy arrays

    IMPORTANT:
    The folder and file ordering is deterministic and matches
    derive_labels_fast() in the training code.
    """

    # Get class folders in deterministic order
    folders = sorted(
        [
            f.name
            for f in os.scandir(path)
            if f.is_dir()
        ],
        key=str.lower
    )

    print(f"\nLoading dataset from: {path}")
    print("Folder order:")

    for class_id, folder in enumerate(folders):
        print(f"  {class_id} -> {folder}")

    # Collect image paths in exactly the same order used by labels
    files = []

    for folder in folders:
        folder_path = os.path.join(
            path,
            folder
        )

        folder_files = get_image_files(
            folder_path
        )

        for file_entry in folder_files:
            files.append(file_entry.path)

    print(f"\nFound {len(files)} image files.")

    # Allocate output array
    data = np.empty(
        (
            len(files),
            224,
            224,
            3
        ),
        dtype=np.uint8
    )

    n = 0
    for i, img_path in enumerate(files):
        img_arr = cv2.imread(img_path)
        if img_arr is None:
            print(
                f"WARNING: Could not read image:\n"
                f"  {img_path}"
            )
            continue

        # IMPORTANT FIX:
        # cv2.imread() returns BGR.
        # PyTorch/PIL expect RGB.
        img_arr = cv2.cvtColor(
            img_arr,
            cv2.COLOR_BGR2RGB
        )

        # Resize to model input size
        img_arr = cv2.resize(
            img_arr,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )
        data[n] = img_arr
        n += 1
    data = data[:n]
    print(
        f"Successfully loaded {len(data)} images."
    )
    return data


# Cache management
def _get_cache_paths(path_needed):
    """
    Returns the cache and version-marker paths.
    """
    os.makedirs(
        CACHE_DIR,
        exist_ok=True
    )
    cache_path = os.path.join(
        CACHE_DIR,
        path_needed
    )
    version_path = cache_path + ".version"
    return cache_path, version_path


def _cache_is_valid(cache_path, version_path):
    """
    Checks whether the existing cache was created using
    the current cache version.
    """
    if not os.path.exists(cache_path):
        return False

    if not os.path.exists(version_path):
        return False

    try:
        with open(
            version_path,
            "r",
            encoding="utf-8"
        ) as f:
            version = f.read().strip()
        return version == CACHE_VERSION
    except Exception:
        return False


def _write_cache_version(version_path):
    """
    Writes the cache version marker.
    """
    with open(
        version_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(CACHE_VERSION)


# Public data loading function
def get_data(path_needed):
    """
    Example:

        get_data("train_x.npy")
        get_data("val_x.npy")

    The function automatically determines whether the cache
    is up to date.

    Old BGR caches will automatically be rebuilt because they
    don't have the current RGB_V2 version marker.
    """

    print(
        f"\nCurrently checking {path_needed}"
    )

    cache_path, version_path = _get_cache_paths(
        path_needed
    )

    # Use cache if it was created using the current pipeline
    if _cache_is_valid(
        cache_path,
        version_path
    ):

        print(
            f"Loading valid cached data:\n"
            f"  {cache_path}"
        )
        return np.load(
            cache_path
        )

    # Cache missing or outdated
    if os.path.exists(cache_path):
        print(
            "Existing cache is outdated."
        )
        print(
            "It will be rebuilt using the RGB pipeline."
        )
        try:
            os.remove(cache_path)
        except OSError:
            pass

    if os.path.exists(version_path):

        try:
            os.remove(version_path)
        except OSError:
            pass

    print(
        "No valid cache found."
    )

    print(
        "Reading images from disk..."
    )

    # Determine source split
    # train_x.npy -> train
    # val_x.npy   -> val
    filename_without_ext = os.path.splitext(
        path_needed
    )[0]
    split_name = filename_without_ext.split(
        "_",
        1
    )[0]
    source_path = os.path.join(
        "archive",
        "animals",
        split_name
    )
    if not os.path.isdir(source_path):
        raise FileNotFoundError(
            f"Dataset directory not found:\n"
            f"  {source_path}"
        )

    # Load images
    x = load_path(
        source_path
    )
    # Save cache
    np.save(
        cache_path,
        x
    )
    _write_cache_version(
        version_path
    )
    print(
        f"\nSaved cached data to:"
        f"\n  {cache_path}"
    )
    print(
        f"Cache version:"
        f" {CACHE_VERSION}"
    )
    print(
        f"Shape:"
        f" {x.shape}"
    )
    del x
    # Reload from disk
    return np.load(
        cache_path
    )