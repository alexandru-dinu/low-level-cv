import matplotlib.pylab as plt
import numpy as np


def gaussian(sigma=1, size=3):
	# range [-k, k] from size
	k = size // 2
	ps = [np.exp(-(z * z / (2 * sigma ** 2))) for z in range(-k, k + 1)]
	f = np.outer(ps, ps)
	f = f / f.sum()
	return f


def box(size=3):
	f = np.ones((size, size)) / size ** 2
	return f


def sobel():
	sx = np.array([
		[-1, 0, 1],
		[-2, 0, 2],
		[-1, 0, 1]
	])

	sy = np.array([
		[1, 2, 1],
		[0, 0, 0],
		[-1, -2, -1]
	])

	return sx, sy


def show_kernel(kernel):
	plt.imshow(kernel)
	plt.colorbar()
	plt.show()
