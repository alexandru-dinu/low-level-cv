import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

np.set_printoptions(threshold=np.nan, linewidth=240)


def show(arr):
	plt.imshow(arr, cmap='gray')
	plt.tight_layout()
	plt.show()


def hsv_thresholding(hsv, low_thr, high_thr):
	"""
	HSV -> Binary image
	"""
	mask = cv2.inRange(
		hsv, np.array(args.low_thr), np.array(args.high_thr)
	)

	assert mask[mask == 255].size + mask[mask == 0].size == mask.size

	if False:
		strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		# mask = cv2.medianBlur(mask, ksize=7)
		mask = cv2.erode(mask, strel, iterations=2)
		mask = cv2.dilate(mask, strel, iterations=3)

	return mask


def noisy_cc_remove(img, num_iter=5):
	"""
	Binary image -> Binary image
	"""
	strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

	out = img.copy()
	out = cv2.erode(out, strel, iterations=1)
	out = cv2.dilate(out, strel, iterations=2)

	for _ in range(num_iter):
		out = cv2.dilate(out, strel, iterations=3)
		out = cv2.erode(out, strel, iterations=3)

	return out


def blob_detection(img):
	"""
	Binary image -> [blob]
	"""
	h, w = img.shape

	lbl = 2
	blobs = []

	# for y in range(h):
	# 	for x in range(w):
	# 		if img[y, x] != 1:
	# 			continue

	# 		num, im, msk, rect = cv2.floodFill(img, None, (y+20, x), lbl)
	# 		ry, rx, rh, rw = rect

	# 		blob = []

	# 		for i in range(ry, ry + rh):
	# 			for j in range(rx, rx + rw):
	# 				if (img[i, j] != lbl):
	# 					continue
	# 				blob.append((i, j))

	# 		blobs.append(blob)
	# 		lbl += 1

	# return blobs


def ellipse_fitting(img, blobs):
	"""
	for each blob
		l1, l2 = eigenvalues(blob)
		if 1 <= l1/l2 <= 2
			fit & draw ellipse
	"""

	for x in blobs:
		xp = np.matrix(x - x.mean(axis=0))
		s = (1 / (len(xp) - 1)) * xp.T * xp
		evalues, evectors = np.linalg.eig(s)
		m_evalues = evalues + x.mean(axis=0)
		return m_evalues


def process_single(args):
	img = cv2.imread(args.img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# --low_thr 0 50 150 --high_thr 25 255 255
	mask = hsv_thresholding(hsv, args.low_thr, args.high_thr)
	# mask = np.divide(mask, 255).astype(np.uint8)

	out = noisy_cc_remove(mask)

	show(np.vstack((out, mask)))


def main(args):
	process_single(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', required=True, type=str)
	parser.add_argument('--low_thr', required=True, nargs='+', type=int)
	parser.add_argument('--high_thr', required=True, nargs='+', type=int)
	parser.add_argument('--debug', action='store_true')
	args = parser.parse_args()

	main(args)
