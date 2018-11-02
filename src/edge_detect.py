import argparse

from kernels import sobel_kernels, gaussian_kernel
from filtering import filter_image
from utils import *


class GradientOrientation:
	E_W = 0
	NE_SW = 1
	N_S = 2
	NW_SE = 3


CANNY_STRONG = 255
CANNY_WEAK = 100


def sobel(img):
	sx, sy = sobel_kernels()
	gx, gy = filter_image(img, [sx, sy], {'pad': 'symmetric'})

	gx, gy = gx[:, :, 0], gy[:, :, 0]

	gradient = np.hypot(gx, gy)
	angles = np.arctan2(gy, gx)

	return gradient, gx, gy, angles


def compute_orientations(angles):
	"""
	Given the set of angles from sobel, map each angle to its orientation
	"""

	h, w = angles.shape
	deg = np.rad2deg(angles) % 180

	orientations = np.zeros((h, w), dtype=np.uint8)

	orientations[(deg < 22.5) | (157.5 <= deg)] = GradientOrientation.E_W
	orientations[(22.5 <= deg) & (deg < 67.5)] = GradientOrientation.NE_SW
	orientations[(67.5 <= deg) & (deg < 112.5)] = GradientOrientation.N_S
	orientations[(112.5 <= deg) & (deg < 157.5)] = GradientOrientation.NW_SE

	return orientations


def thresholding(img, low_thr, high_thr):
	s_i, s_j = np.where(img > high_thr)
	w_i, w_j = np.where((img >= low_thr) & (img <= high_thr))
	r_i, r_j = np.where(img < low_thr)

	img[s_i, s_j] = CANNY_STRONG
	img[w_i, w_j] = CANNY_WEAK
	img[r_i, r_j] = 0

	return img


def edge_tracking(img):
	"""
	Mark weak pixels as strong if they are connected
	to at least one strong neighbouring pixel
	"""

	h, w = img.shape[:2]

	for y in range(1, h - 1):
		for x in range(1, w - 1):
			neighbours = [
				img[y - 1][x],
				img[y + 1][x],
				img[y][x - 1],
				img[y][x + 1],
				img[y - 1][x - 1],
				img[y + 1][x + 1],
				img[y + 1][x - 1],
				img[y - 1][x + 1]
			]

			strong_neighbours = map(lambda p: p == CANNY_STRONG, neighbours)

			if img[y][x] == CANNY_WEAK:
				img[y][x] = CANNY_STRONG if any(strong_neighbours) else 0

	return img


def non_max_suppression(gradient, orientations):
	"""
	Look for local maxima w.r.t. gradient's angles.
	Keep only the local maxima in the resulting suppressed gradient.
	"""

	h, w = gradient.shape
	nms_grad = np.zeros((h, w), dtype=np.float32)

	for y in range(1, h - 1):
		for x in range(1, w - 1):
			o = orientations[y, x]
			g = gradient[y, x]

			# E->W gradient => N->S edge
			if o == GradientOrientation.E_W:
				if g >= gradient[y, x - 1] and g >= gradient[y, x + 1]:
					nms_grad[y - 1, x - 1] = g

			# NE->SW gradient => NW->SE edge
			elif o == GradientOrientation.NE_SW:
				if g >= gradient[y - 1, x + 1] and g >= gradient[y + 1, x - 1]:
					nms_grad[y - 1, x - 1] = g

			# N->S gradient => E->W edge
			elif o == GradientOrientation.N_S:
				if g >= gradient[y - 1, x] and g >= gradient[y + 1, x]:
					nms_grad[y - 1, x - 1] = g

			# NW->SE gradient -> NE->SW
			elif o == GradientOrientation.NW_SE:
				if g >= gradient[y - 1, x - 1] and g >= gradient[y + 1, x + 1]:
					nms_grad[y - 1, x - 1] = g

	return nms_grad


def canny(img, low_thr=100, high_thr=200, sigma=0.75):
	"""
	Perform canny edge detection
	img -> gaussian filtering -> sobel filtering -> nms -> thresholding -> edge tracking
	"""

	img_smooth = filter_image(img, gaussian_kernel(sigma=sigma, size=5), {'pad': 'symmetric'})

	gradient, gx, gy, angles = sobel(img_smooth)

	orientations = compute_orientations(angles)

	out = non_max_suppression(gradient, orientations)

	out = thresholding(out, low_thr, high_thr)

	out = edge_tracking(out)

	return out


def main(args):
	img = open_img(args.img_path, mode='gray')
	out = canny(img, args.low_thr, args.high_thr, args.sigma)

	show_side_by_side(img, out, title2='edges')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', type=str)
	parser.add_argument('--low_thr', type=int)
	parser.add_argument('--high_thr', type=int)
	parser.add_argument('--sigma', type=float)

	main(parser.parse_args())
