import argparse

import cv2
import numpy as np

from utils import show_histogram, get_ellipse, show

np.set_printoptions(threshold=np.nan, linewidth=240)


def determine_hsv_ranges(histogram):
	# TODO
	pass


def hsv_thresholding(hsv, low_thr, high_thr):
	"""
	HSV -> Binary image
	"""
	mask = cv2.inRange(
		hsv, np.array(low_thr), np.array(high_thr)
	)

	assert mask[mask == 255].size + mask[mask == 0].size == mask.size

	return mask


def noisy_cc_remove(img, num_iter=5):
	"""
	Binary image -> Binary image
	"""
	strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

	out = img.copy()

	for _ in range(num_iter):
		out = cv2.dilate(out, strel, iterations=2)
		out = cv2.erode(out, strel, iterations=1)

	out = cv2.erode(out, strel, iterations=1)

	return out


def blob_detection(img, thr):
	"""
	Binary image -> [blob]
	"""
	cimg = img.copy()

	h, w = cimg.shape

	lbl = 2
	blobs = []
	rects = []

	for y in range(h):
		for x in range(w):
			if cimg[y, x] != 1:
				continue

			_, _, _, rect = cv2.floodFill(cimg, mask=None, seedPoint=(x, y), newVal=lbl)
			ry, rx, rh, rw = rect

			blob = []

			for i in range(ry, ry + rh):
				for j in range(rx, rx + rw):
					if cimg[j, i] != lbl:
						continue
					blob.append((i, j))

			if len(blob) < thr:
				print(f"Rejecting blob with len {len(blob)}")
				continue

			blobs.append(np.array(blob))
			rects.append(((ry, rx), (ry + rh, rx + rw)))
			lbl += 1

	return blobs, rects


def ellipse_fitting(blobs):
	ellipses = []

	for blob in blobs:
		blob = np.array(blob)
		ellipse = get_ellipse(blob)
		ellipses.append(ellipse)

	return ellipses


def process_single(args):
	img = cv2.imread(args.img_path)

	h, w, d = img.shape

	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	if args.hist:
		show_histogram(hsv, lthr=0, hthr=70, bin_count=35)
		exit(0)

	# --low_thr 0 50 100 --high_thr 25 150 255
	mask = hsv_thresholding(hsv, args.low_thr, args.high_thr)

	out = noisy_cc_remove(mask, num_iter=3)
	out = np.divide(out, 255).astype(np.uint8)

	blobs, rects = blob_detection(out, args.blob_thr)

	# show filtered blobs
	filtered = np.zeros((h, w), dtype=np.uint8)
	for blob in blobs:
		for x, y in blob:
			filtered[y, x] = 1

	# for (tl, br) in rects[:1]:
	# 	cv2.rectangle(xx, tl, br, 255, 2)

	# fit ellipses
	for blob, rect in zip(blobs, rects):
		ellipse = cv2.fitEllipse(blob)
		print(ellipse)

		ellipse = get_ellipse(blob, 4)
		print(ellipse)
		print("---")

		# exit(0)
		# area = cv2.contourArea(blob)
		# perim = cv2.arcLength(blob, closed=True)
		# x = (np.pi * 4 * area) / (perim ** 2)
		# if x > 0.51:
		# 	print(f"Rejecting blob with x = {x}")
		# 	continue

		cv2.ellipse(img, ellipse, (0, 255, 0), 2)

	show(filtered, cmap='gray')
	show(img, from_cv2=True)

	pass


def main(args):
	process_single(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', required=True, type=str)
	parser.add_argument('--hist', action='store_true')
	parser.add_argument('--blob_thr', type=int)
	parser.add_argument('--low_thr', nargs='+', type=int)
	parser.add_argument('--high_thr', nargs='+', type=int)
	parser.add_argument('--debug', action='store_true')
	args = parser.parse_args()

	main(args)
