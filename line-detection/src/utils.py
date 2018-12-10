import cv2
import numpy as np
import matplotlib.pyplot as plt


def open_img(img_path, gray=False):
	if gray:
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	else:
		img = cv2.imread(img_path)

	return img


def show(arr, from_cv2=False, cmap=None):
	if from_cv2:
		b = arr[:, :, 0]
		g = arr[:, :, 1]
		r = arr[:, :, 2]
		arr = np.dstack((r, g, b))

	plt.imshow(arr, cmap=cmap)
	plt.tight_layout()

	plt.show()
