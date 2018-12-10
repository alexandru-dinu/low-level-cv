import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

np.set_printoptions(suppress=True)


def generate(num_inliers=200, num_outliers=25):
	h, w = 300, 700
	xs = np.zeros((h, w), dtype=np.uint8)

	y = lambda x: 3 * x - 20

	ns = np.linspace(1, num_inliers, num=num_inliers, dtype=np.uint8)

	ox = 33
	oy = 20

	for n in ns:
		rx = np.random.randint(20, dtype=np.uint8)
		ry = np.random.randint(30, dtype=np.uint8)
		xs[n + ox + rx, y(n) + oy + ry] = 1

	for _ in range(num_outliers):
		cx, cy = ns[ns.size // 2], y(ns[ns.size // 2])
		rx = np.random.randint(0, 2 * cx)
		ry = np.random.randint(0, 2 * cy)
		xs[rx, ry] = 1

	return xs


def fit_lsq(x, y):
	"""
	return linear model: a*x + b
	"""

	x = np.matrix(np.vstack((x, np.ones(len(x))))).T
	y = np.matrix(y).T

	model = np.linalg.inv(x.T * x) * x.T * y

	a = model[0, 0]
	b = model[1, 0]

	return model, lambda p: a * p + b


def dist(model, x, y):
	return np.abs(y - model(x))


def ransac(data, sample_size=10, num_iter=100, inlier_thr=10, target_inliers=None):
	"""
	"""

	x, y = np.where(data == 1)
	z = list(zip(x, y))

	# plt.scatter(x, y)
	# plt.show()

	best_inlier_count = 0
	best_model = None

	for it in range(num_iter):
		sample = random.sample(z, sample_size)

		sx = np.array([e[0] for e in sample])
		sy = np.array([e[1] for e in sample])

		model_vec, model = fit_lsq(sx, sy)
		print(f"[{it}] -> {model_vec.T}")

		inlier_count = sum([1 for (x, y) in zip(x, y) if dist(model, x, y) <= inlier_thr])

		if inlier_count > best_inlier_count:
			best_inlier_count = inlier_count
			best_model = (model_vec, model)

			if target_inliers is not None and inlier_count > target_inliers:
				break

	print(f"[+] Took {it} iterations. Best inlier count = {best_inlier_count}.")

	return best_model


if __name__ == '__main__':
	data = generate()

	model_vec, model = ransac(data, sample_size=20, num_iter=2000, inlier_thr=5)
	print(colored(f"Best model: f{model_vec.T}", "green"))

	# img = cv2.cvtColor(data.copy(), cv2.COLOR_GRAY2RGB)
	img = data.copy()

	x1, x2 = 0, 500
	y1, y2 = int(model(x1)), int(model(x2))
	cv2.line(img, (y1, x1), (y2, x2), 1, thickness=1)

	plt.imshow(img, cmap='gray')
	plt.show()

	pass
