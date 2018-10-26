import argparse

from kernels import sobel_kernels, gaussian_kernel
from filtering import do_filtering
from utils import *


class Orientation:
	E_W = 0
	NE_SW = 1
	N_S = 2
	NW_SE = 3


CANNY_STRONG = 255
CANNY_WEAK = 100


def sobel(img):
	sx, sy = sobel_kernels()
	gx, gy = do_filtering(img, [sx, sy], {'pad': 'symmetric'})

	gx, gy = gx[:, :, 0], gy[:, :, 0]

	gradient = np.hypot(gx, gy)
	orientation = np.arctan2(gy, gx)

	return gradient, gx, gy, orientation


def compute_directions(orientation):
	"""
	Given the set of angles from sobel, map each angle to the direction it gives
	"""

	h, w = orientation.shape
	deg = np.rad2deg(orientation) % 180

	dirs = np.zeros((h, w), dtype=np.uint8)

	# E->W (horizontal gradient)
	dirs[(deg < 22.5) | (157.5 <= deg)] = Orientation.E_W
	# NE->SW
	dirs[(22.5 <= deg) & (deg < 67.5)] = Orientation.NE_SW
	# N->S (vertical gradient)
	dirs[(67.5 <= deg) & (deg < 112.5)] = Orientation.N_S
	# NW->SE
	dirs[(112.5 <= deg) & (deg < 157.5)] = Orientation.NW_SE

	return dirs


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
	Mark weak pixels as strong if they are connected to a strong pixel
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

			# if there is a weak pixel connected to a strong pixel,
			# mark it as strong
			if img[y][x] == CANNY_WEAK:
				img[y][x] = CANNY_STRONG if any(strong_neighbours) else 0

	return img


def compute_suppressed_gradient(gradient, dirs):
	"""
	Look for local maxima w.r.t. gradient's orientation.
	Keep only the local maxima in the resulting suppressed gradient.
	"""

	h, w = gradient.shape
	sup_grad = np.zeros((h, w), dtype=np.float32)

	for y in range(1, h - 1):
		for x in range(1, w - 1):
			o = dirs[y, x]
			g = gradient[y, x]

			# horizontal gradient => check E/W
			if o == Orientation.E_W:
				if g >= gradient[y, x - 1] and g >= gradient[y, x + 1]:
					sup_grad[y - 1, x - 1] = g

			# "second diagonal" gradient
			elif o == Orientation.NE_SW:
				if g >= gradient[y - 1, x + 1] and g >= gradient[y + 1, x - 1]:
					sup_grad[y - 1, x - 1] = g

			# vertical gradient => check N/S
			elif o == Orientation.N_S:
				if g >= gradient[y - 1, x] and g >= gradient[y + 1, x]:
					sup_grad[y - 1, x - 1] = g

			# "first diagonal" gradient
			elif o == Orientation.NW_SE:
				if g >= gradient[y - 1, x - 1] and g >= gradient[y + 1, x + 1]:
					sup_grad[y - 1, x - 1] = g

	return sup_grad


def canny(img, low_thr, high_thr, sigma=0.75):
	img_smooth = do_filtering(img, [gaussian_kernel(sigma=sigma, size=5)], {'pad': 'symmetric'})

	gradient, gx, gy, orientation = sobel(img_smooth[0])

	dirs = compute_directions(orientation)

	out = compute_suppressed_gradient(gradient, dirs)

	out = thresholding(out, low_thr, high_thr)

	out = edge_tracking(out)

	return np.expand_dims(out, axis=2)


def main(args):
	img = open_img(args.img_path, args.img_mode)
	out = canny(img, args.low_thr, args.high_thr, args.sigma)

	show_side_by_side(img, out, title2='edges')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', type=str)
	parser.add_argument('--img_mode', type=str)
	parser.add_argument('--low_thr', type=int)
	parser.add_argument('--high_thr', type=int)
	parser.add_argument('--sigma', type=float)

	main(parser.parse_args())
