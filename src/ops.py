import numpy as np


def conv(pimg, kernel, pad):
	ph, pw, chs = pimg.shape

	out = np.zeros((ph - 2 * pad, pw - 2 * pad, chs), dtype=np.float32)

	for c in range(chs):
		for y in range(pad, ph - pad):
			for x in range(pad, pw - pad):
				patch = pimg[y - pad:y + pad + 1, x - pad:x + pad + 1, c]
				out[y - pad, x - pad, c] = np.sum(patch * kernel) / 255.0

	return out


def do_filtering(img, kernels, settings):
	"""
	settings = {'pad'}
	"""
	assert type(kernels) == list, "kernels are not in a list"
	assert settings['pad'] in ['symmetric', 'zeros']

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

		out[ki] = conv(pimg, kernel, p)

	return out

# def do_edge_detect(img, kernels, settings)
