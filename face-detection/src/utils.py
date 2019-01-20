import argparse
import json
import urllib.request as req
from random import shuffle

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import ImageDraw


def download_images(count=10):
	data = [json.loads(x.strip()) for x in open("../data/face_detection.json")]
	shuffle(data)

	for i in range(count):
		img = open(f"../data/img_{i}.jpg", 'wb')
		img.write(req.urlopen(data[i]['content']).read())
		img.close()

		print(f"Downloaded ../data/img_{i}.jpg")


def show(arr, from_cv2=False, cmap=None):
	if from_cv2:
		b = arr[:, :, 0]
		g = arr[:, :, 1]
		r = arr[:, :, 2]
		arr = np.dstack((r, g, b))

	plt.imshow(arr, cmap=cmap)
	plt.tight_layout()

	plt.show()


def morph_operation(img, mode, strel, num_iter=1):
	func = {'erode': np.min, 'dilate': np.max}

	assert mode in func.keys(), "Invalid mode"
	assert img.ndim == 2, "Only 2D binary image"
	assert strel.ndim == 2, "Only 2D strel"

	pad_value = 0 if mode == 'dilate' else 1

	h, w = img.shape
	k, _ = strel.shape

	p = (k - 1) // 2
	pad = ((p, p), (p, p))
	pimg = np.pad(img, pad, mode='constant', constant_values=pad_value)

	out = None

	for i in range(num_iter):
		out = np.zeros((h, w), dtype=img.dtype)

		for y in range(p, h + p):
			for x in range(p, w + p):
				patch = np.ma.array(pimg[y - p:y + p + 1, x - p:x + p + 1], mask=1 - strel)

				out[y - p, x - p] = func[mode](patch)

		pimg = np.pad(out.copy(), pad, mode='constant', constant_values=pad_value)

	return out


def unit_vector(vector):
	return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)

	return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def eig_sorted(mat):
	eig_vals, eig_vecs = np.linalg.eig(mat)
	idx = np.argsort(eig_vals)[::-1]
	return eig_vals[idx], eig_vecs[:, idx]


def get_ellipse(xs, sf=np.sqrt(2)):
	"""
	xs (blob) -> fitting ellipse
	"""
	assert xs.ndim == 2 and xs.shape[1] == 2
	N = xs.shape[0]
	mean = xs.mean(axis=0)

	xp = np.matrix(xs - mean)
	cov = (1 / (N - 1)) * xp.T * xp
	eig_values, eig_vectors = eig_sorted(cov)

	a = np.rad2deg(np.arccos(np.clip(eig_vectors[0, 1], -1.0, 1.0)))

	eig_vectors[:, 0] *= np.sqrt(eig_values[0]) * sf
	eig_vectors[:, 1] *= np.sqrt(eig_values[1]) * sf
	eig_vectors += mean

	v1 = np.array([eig_vectors[0, 0], eig_vectors[1, 0]])
	v2 = np.array([eig_vectors[0, 1], eig_vectors[1, 1]])

	(cx, cy) = mean
	(MA, ma) = np.sqrt(eig_values) * sf

	return (cx, cy), (ma, MA), a


def show_histogram(img, lthr=5, hthr=30, bin_count=50):
	hue, sat, val = cv2.split(img)

	roi = np.where((lthr <= hue) & (hue <= hthr))
	hue = hue[roi]
	sat = sat[roi]
	val = val[roi]

	plt.subplot(1, 3, 1)
	n, bins, patches = plt.hist(hue, bin_count)
	plt.title("Hue")

	plt.subplot(1, 3, 2)
	n, bins, patches = plt.hist(sat, bin_count)
	plt.title("Sat")

	plt.subplot(1, 3, 3)
	n, bins, patches = plt.hist(val, bin_count)
	plt.title("Val")

	plt.tight_layout()
	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--download', action='store_true')
	parser.add_argument('--num', type=int)
	args = parser.parse_args()

	if args.download:
		download_images(args.num)
