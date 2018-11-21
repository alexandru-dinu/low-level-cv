import json
import urllib.request as req
from random import shuffle
import argparse
import numpy as np


def download_images(count=10):
	data = [json.loads(x.strip()) for x in open("../data/face_detection.json")]
	shuffle(data)

	for i in range(count):
		img = open(f"../data/img_{i}.jpg", 'wb')
		img.write(req.urlopen(data[i]['content']).read())
		img.close()

		print(f"Downloaded ../data/img_{i}.jpg")


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

	for i in range(num_iter):
		out = np.zeros((h, w), dtype=img.dtype)

		for y in range(p, h + p):
			for x in range(p, w + p):
				patch = pimg[y - p:y + p + 1, x - p:x + p + 1]
				out[y - p, x - p] = func[mode](patch * strel)

		pimg = np.pad(out.copy(), pad, mode='constant', constant_values=pad_value)

	return out


def get_ellipse(xs, f=np.sqrt(2)):
	"""
	xs (blob) -> fitting ellipse
	"""
	assert xs.ndim == 2 and xs.shape[1] == 2
	N = xs.shape[0]
	mean = xs.mean(axis=0)

	xp = np.matrix(xs - mean)
	cov = (1 / (N - 1)) * xp.T * xp
	eig_values, eig_vectors = np.linalg.eig(cov)
	eig_vectors += mean

	eig_vectors[:, 0] *= np.sqrt(eig_values[0]) * f
	eig_vectors[:, 1] *= np.sqrt(eig_values[1]) * f

	v1, v2 = eig_vectors[:, 0], eig_vectors[:, 1]

	t = np.linspace(0, 2 * np.pi, num=N, endpoint=False)
	ellipse = np.zeros_like(xs)
	ellipse[:, 0] = v1[0] * np.cos(t) + v2[0] * np.sin(t)
	ellipse[:, 1] = v1[1] * np.cos(t) + v2[1] * np.sin(t)

	return ellipse


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--download', action='store_true')
	parser.add_argument('--num', type=int)
	args = parser.parse_args()

	if args.download:
		download_images(args.num)
