import argparse

from edge_detect import canny
from utils import *


def generate_template(img):
	tpl = canny(img, low_thr=180, high_thr=230, sigma=1.1)
	return tpl


def match_template(img_edges, tpl_edges):
	scales = [0.4, 0.6, 0.8, 1.4]
	tpl_ms = [
		cv2.resize(tpl_edges, None, fx=i, fy=i) for i in scales
	]

	best_loc, best_val, best_scale = None, -np.inf, None

	for si, tpl in enumerate(tpl_ms):
		th, tw = tpl.shape[:2]

		res = cv2.matchTemplate(img_edges, tpl, cv2.TM_CCOEFF)
		_, max_val, _, max_loc = cv2.minMaxLoc(res)

		if max_val > best_val:
			best_val = max_val
			best_loc = max_loc
			best_scale = si
		if max_val < best_val:
			break

		top = best_loc
		bot = (top[0] + tw, top[1] + th)

	from termcolor import colored
	print(colored(f"Detected at scale {scales[best_scale]}", 'green'))

	return top, bot


def main(args):
	gimg = open_img(args.img_path, 'gray')
	cimg = open_img(args.img_path, 'rgb')
	tpl = open_img(args.tpl_path, 'gray')

	tpl_edges = generate_template(tpl)
	img_edges = canny(gimg, low_thr=70, high_thr=120, sigma=1.2)

	out = match_template(img_edges, tpl_edges)

	top, bot = out
	cv2.rectangle(cimg, top, bot, (0, 255, 0), 3)

	show_img(cimg)
	pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', type=str)
	parser.add_argument('--tpl_path', type=str)
	# parser.add_argument('--box_size', type=int)

	main(parser.parse_args())
