import argparse

from kernels import box_kernel, gaussian_kernel
from utils import *


def conv(pimg, kernel, pad, normalize=False):
	ph, pw, chs = pimg.shape

	out = np.zeros((ph - 2 * pad, pw - 2 * pad, chs), dtype=np.float32)

	for c in range(chs):
		for y in range(pad, ph - pad):
			for x in range(pad, pw - pad):
				patch = pimg[y - pad:y + pad + 1, x - pad:x + pad + 1, c]
				out[y - pad, x - pad, c] = np.sum(patch * kernel)

	return out / 255.0 if normalize else out


def strided_conv(img, kernel):
	h, w, chs = img.shape
	k, _ = kernel.shape
	ph, pw = h + h % k, w + w % k

	f = lambda x: (x // k, x - x // k)

	img = np.pad(img, (f(ph), f(pw), (0, 0)), mode='symmetric')

	out = np.zeros((ph // k, pw // k, chs), dtype=np.float32)

	for c in range(chs):
		for y in range(0, ph // k):
			for x in range(0, pw // k):
				patch = img[y * k:(y + 1) * k, x * k:(x + 1) * k, c]
				out[y, x, c] = np.sum(patch * kernel)

	return out


def median_filtering(img, size, normalize=False):
	k = size
	p = (k - 1) // 2  # pad size
	pad = ((p, p), (p, p), (0, 0))  # h, w, c
	pimg = np.pad(img, pad, mode='symmetric')

	ph, pw, chs = pimg.shape

	out = np.zeros((ph - 2 * p, pw - 2 * p, chs), dtype=np.float32)

	for c in range(chs):
		for y in range(p, ph - p):
			for x in range(p, pw - p):
				patch = pimg[y - p:y + p + 1, x - p:x + p + 1, c]
				out[y - p, x - p, c] = np.median(patch)

	return out / 255.0 if normalize else out


def filter_image(img, kernels, settings):
	assert settings['pad'] in ['symmetric', 'zeros']
	normalize = settings.get('normalize', False)

	if type(kernels) != list:
		kernels = [kernels]

	out = [None] * len(kernels)

	# perform filtering with each kernel
	for ki, kernel in enumerate(kernels):
		k, _ = kernel.shape

		p = (k - 1) // 2  # pad size
		pad = ((p, p), (p, p), (0, 0))  # h, w, c

		if settings['pad'] == 'zeros':
			pimg = np.pad(img, pad, mode='constant', constant_values=0)
		if settings['pad'] == 'symmetric':
			pimg = np.pad(img, pad, mode='symmetric')

		out[ki] = conv(pimg, kernel, p, normalize)

	if len(kernels) == 1:
		out = out[0]

	return out


def main(args):
	assert args.filter in ['gaussian', 'box']

	img = open_img(args.img_path, args.img_mode)

	if args.filter == 'gaussian':
		kern = gaussian_kernel(sigma=args.sigma, size=args.size)
		out = filter_image(img, kern, settings={'pad': 'symmetric', 'normalize': True})
		print(np.min(out), np.max(out))

	if args.filter == 'box':
		kern = box_kernel(size=args.size)
		out = filter_image(img, kern, settings={'pad': 'symmetric', 'normalize': True})

	# show_kernel(kern)
	show_side_by_side(img, out)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', type=str)
	parser.add_argument('--img_mode', type=str)
	parser.add_argument('--filter', type=str)
	parser.add_argument('--size', type=int)
	parser.add_argument('--sigma', type=float)

	main(parser.parse_args())
