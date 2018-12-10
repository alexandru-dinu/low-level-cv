import argparse

import cv2

from utils import open_img, show


def get_edges(img):
	out = img.copy()

	out = cv2.GaussianBlur(out, ksize=(3, 3), sigmaX=0)
	out = cv2.Canny(out, 200, 230)

	return out


def morphological(img):
	out = img.copy()

	strel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(7, 7))

	out = cv2.dilate(out, strel, iterations=1)
	out = cv2.erode(out, strel, iterations=1)

	return out


def main(args):
	img = open_img(args.img_path)
	gray = open_img(args.img_path, gray=True)

	# show(img, from_cv2=True)

	out = get_edges(gray)
	out = morphological(out)

	show(out, cmap='gray')

	pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', required=True, type=str)
	main(parser.parse_args())
