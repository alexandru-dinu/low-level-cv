import cv2
import matplotlib.pylab as plt
import numpy as np


def open_img(img_path, mode='rgb'):
	# h, w, c :: bgr->rgb
	if mode == 'gray':
		img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
		img = np.expand_dims(img, axis=2)
	else:  # mode == 'rgb'
		img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

	return img


def show_img(img, colorbar=False):
	if img.ndim == 2:
		img = np.expand_dims(img, axis=2)

	if img.shape[2] == 1:
		plt.imshow(img[:, :, 0], cmap='gray')
	else:
		plt.imshow(img)

	if colorbar:
		plt.colorbar()

	plt.show()


def show_kernel(kernel):
	plt.imshow(kernel)
	plt.colorbar()
	plt.show()


def show_side_by_side(img1, img2, title1='original', title2='filtered'):
	f, ax = plt.subplots(1, 2, figsize=(25, 50))

	ax[0].title.set_text(title1)
	ax[1].title.set_text(title2)

	if img1.ndim == 2:
		img1 = np.expand_dims(img1, axis=2)
	if img2.ndim == 2:
		img2 = np.expand_dims(img2, axis=2)

	if img1.shape[2] == 1:
		ax[0].imshow(img1[:, :, 0], cmap='gray')
	else:
		ax[0].imshow(img1)

	if img2.shape[2] == 1:
		ax[1].imshow(img2[:, :, 0], cmap='gray')
	else:
		ax[1].imshow(img2)

	plt.tight_layout()
	plt.show()


def show_channel(img, ch):
	chs = {'r': 0, 'g': 1, 'b': 2}
	assert ch in chs
	assert img.ndim == 3

	iimg = img.copy()
	h, w, _ = iimg.shape

	for c in chs.keys() - {ch}:
		iimg[:, :, chs[c]] = np.zeros((h, w))

	show_img(iimg)
