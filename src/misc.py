import argparse
from termcolor import colored

from filtering import strided_conv, filter_image, median_filtering
from kernels import gaussian_kernel, box_kernel
from utils import *
from operator import itemgetter


def hsv_red_contours(args):
	orig_img = open_img(args.img_path)

	img = cv2.imread(args.img_path)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	mask1 = cv2.inRange(
		hsv, np.array([0, 110, 100]), np.array([10, 255, 255])
	)
	mask2 = cv2.inRange(
		hsv, np.array([170, 110, 100]), np.array([180, 255, 255])
	)

	mask = (mask1 | mask2)  # / 255
	assert mask[mask == 255].size + mask[mask == 0].size == mask.size

	# out = median_filtering(np.expand_dims(mask, axis=2), args.size)
	# out = out[:, :, 0].astype(np.uint8)
	out = cv2.medianBlur(mask, args.size)
	_, contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# # draw all contours
	# for i, cont in enumerate(contours):
	# 	bbox = cv2.boundingRect(cont)
	# 	tl, br = (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])
	# 	cv2.rectangle(orig_img, tl, br, color=(0, 255, 0), thickness=3)

	# draw only the largest contour
	bboxes = [cv2.boundingRect(cont) for cont in contours]
	areas = sorted([(b, b[2] * b[3]) for b in bboxes if b[2] * b[3] >= 900], key=itemgetter(1))
	# areas = sorted(areas, key=itemgetter(1))
	# bbox = bboxes[np.argmax(areas)

	print(colored(f"Detected: {len(areas)}", "green"))

	for (bbox, _) in areas:
		tl, br = (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3])
		cv2.rectangle(orig_img, tl, br, color=(0, 255, 0), thickness=3)

	show_img(orig_img)


# TODO: return a list of possible bboxes and perform IOU with matchTemplate results


def color_clusters(args):
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

	# color_clusters(parser.parse_args())
	hsv_red_contours(parser.parse_args())
