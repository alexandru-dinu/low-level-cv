import argparse
from termcolor import colored

from filtering import strided_conv
from kernels import gaussian_kernel, box_kernel
from utils import *


def main(args):
    # assert args.filter in ['gaussian', 'box']

    img = open_img(args.img_path, args.img_mode)
    kern = box_kernel(args.size)

    r = get_channel(img, 'r')
    out = strided_conv(r, kern)
    out_r = out[:, :, 0]

    h, w, _ = out.shape
    z = np.zeros_like(out_r)
    out_r = np.round(np.dstack((out_r, z, z))).astype(np.uint8)

    show_side_by_side(img, out_r, title2='color_clusters')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', type=str)
	parser.add_argument('--img_mode', type=str)
	# parser.add_argument('--filter', type=str)
	parser.add_argument('--size', type=int)

	main(parser.parse_args())
