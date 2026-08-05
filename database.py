import numpy as np
import pandas as pd
import torch
import timm
import os
import cv2
import subprocess
import sys

CACHE = "cached_data.npz"

def get_data():
	if os.path.exists(CACHE):
		print(f"Loading data from cache: {CACHE}")
		data = np.load(CACHE)
		return data["train_x"], data["test_x"], data["val_x"]

	print(f"No cache found, reading images...")
	train_path = "./archive/animals/train"
	test_path = "./archive/animals/test"
	val_path = "./archive/animals/validation"

	def load_path(path):
		data = []
		for folder in os.listdir(path):
			sub_path = os.path.join(path, folder)
			if not os.path.isdir(sub_path):
				continue
			for img in os.listdir(sub_path):
				img_path = os.path.join(sub_path, img)
				img_arr = cv2.imread(img_path)
				if img_arr is not None:
					img_arr = cv2.resize(img_arr, (224, 224))
					data.append(img_arr)
		return np.array(data)

	train_x = load_path(train_path)
	test_x = load_path(test_path)
	val_x = load_path(val_path)

	np.savez(CACHE, train_x=train_x, test_x=test_x, val_x=val_x)
	print(f"Saved binary cache to {CACHE}")
	print(f"Loaded: train={len(train_x)}, test={len(test_x)}, val={len(val_x)}")
	return train_x, test_x, val_x


if __name__ == "__main__":
	get_data()