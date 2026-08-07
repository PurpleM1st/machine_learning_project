import numpy as np
import os
import cv2

CACHE = "cached_data.npz"

def load_path(path):
	files = []
	for folder in os.listdir(path):
		sub_path = os.path.join(path, folder)
		if not os.path.isdir(sub_path):
			continue
		for img in os.listdir(sub_path):
			files.append(os.path.join(sub_path, img))
	data = np.empty((len(files), 224, 224, 3), dtype=np.uint8)
	n = 0
	for img_path in files:
		img_arr = cv2.imread(img_path)
		if img_arr is not None:
			data[n] = cv2.resize(img_arr, (224, 224))
			n += 1
	return data[:n]

def get_data(path_needed):
	print(f"Currently checking {path_needed}")

	os.makedirs("cache", exist_ok=True)

	if os.path.exists(f"cache/{path_needed}"):
		return np.load(f"cache/{path_needed}")

	print(f"No cache found, reading images...")
	path_arr = os.path.splitext(path_needed)[0]
	path_arr = path_arr.split("_", 1)[0]

	x = load_path(f"archive/animals/{path_arr}")
	np.save(f"cache/{path_needed}", x)
	del x

	return np.load(f"cache/{path_needed}")
