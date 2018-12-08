import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse

from utils import show_histogram, get_ellipse

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

	return out


def blob_detection(img):
	"""
	Binary image -> [blob]
	"""
	h, w = img.shape

	lbl = 2
	blobs = []
	rects = []

	for y in range(h):
		for x in range(w):
			if img[y, x] != 1:
				continue

			num, im, msk, rect = cv2.floodFill(img, mask=None, seedPoint=(x, y), newVal=lbl)
			ry, rx, rh, rw = rect

			blob = []

			for i in range(ry, ry + rh):
				for j in range(rx, rx + rw):
					if img[j, i] != lbl:
						continue
					blob.append((i, j))

			blobs.append(blob)
			rects.append(((ry, rx), (ry + rh, rx + rw)))
			lbl += 1

	return blobs, rects


def ellipse_fitting(img, blobs):
	for blob in blobs:
		blob = np.array(blob)
		plt.plot(blob[:, 0], blob[:, 1], '.b', lw=0.2)
		ellipse = get_ellipse(blob)
		plt.plot(ellipse[:, 0], ellipse[:, 1], '-r', lw=1)
		plt.show()
		exit()


def process_single(args):
	img = cv2.imread(args.img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	# show_histogram(hsv, lthr=0, hthr=70, bin_count=20)

	# --low_thr 0 50 150 --high_thr 25 255 255
	mask = hsv_thresholding(hsv, args.low_thr, args.high_thr)

	out = noisy_cc_remove(mask)
	xx = out.copy()
	out = np.divide(out, 255).astype(np.uint8)

	blobs, rects = blob_detection(out)

	ellipse_fitting(xx, blobs)

	for (tl, br) in rects:
		cv2.rectangle(xx, tl, br, 255, 2)
	show(xx)


# show(np.vstack((out, mask)))
# show(out)


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
