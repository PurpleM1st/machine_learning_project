import sys
import cv2
import numpy as np
import joblib
from skimage.feature import hog

def extract_features(img):
    """
    Extracts HOG + HSV features from a single image (array or filepath).
    """
    if isinstance(img, str):
        img = cv2.imread(img)
        if img is None:
            raise FileNotFoundError(f"Could not load image from path: {img}")
            
    img_resized = cv2.resize(img, (128, 128))
    
    # 1. HOG shape features
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

    combined = np.hstack([hog_feat, hsv_feat]).astype(np.float32)
    return combined.reshape(1, -1)

def predict_an_image(img, model_path="models/trained_svm.pkl"):
    """
    Predicts the label for a single image (accepts image array or filepath).
    """
    bundle = joblib.load(model_path)
    scaler = bundle["scaler"]
    svm_model = bundle["svm"]

    features = extract_features(img)
    scaled_features = scaler.transform(features)

    predicted_label = svm_model.predict(scaled_features)[0]
    return predicted_label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Please provide an image path.")
        print("Usage: python3 predict_script.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    label = predict_an_image(image_path)
    print(f"Predicted Label: {label}")