import sys

from kernels import *
from utils import *
from ops import *


def main(args):
	# path, mode(rgb/gray)
	img = open_img(args[1], args[2])

	settings = {
		'pad': 'symmetric',
	}

	# kernel = gaussian(sigma=3, size=13)
	kernel = box(size=5)
	sx, sy = sobel()

	show_kernel(sx)

	exit(1)

	out = do_filtering(img, [sx, kernel], settings)

	show_side_by_side(out[0], out[1])


if __name__ == '__main__':
	main(sys.argv)
