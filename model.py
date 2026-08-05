from database import get_data
import cv2

print("Loading data from cache...")

train_x = get_data("train_x.npy")
print(f"train={len(train_x)} ")
del train_x
test_x = get_data("test_x.npy")
print(f"test={len(test_x)} ")
del test_x
val_x = get_data("val_x.npy")
print(f"val={len(val_x)}")
del val_x