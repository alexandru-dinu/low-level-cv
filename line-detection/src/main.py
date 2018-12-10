import argparse

from utils import *
from ransac import find_sequential


def main(args):
	img = open_img(args.img_path)
	gray = open_img(args.img_path, gray=True)

	edge_img = get_edges(gray)
	edge_img = np.divide(edge_img, 255).astype(np.uint8)

	# show(edge_img, cmap='gray')

	config = {
		'num_lines': 12,
		'sample_size': 10,
		'num_iter': 2000,
		'inlier_thr': 10
	}
	model_vecs, models = find_sequential(edge_img, config)

	for m in models:
		x1, x2 = 0, 200
		y1, y2 = int(m(x1)), int(m(x2))
		cv2.line(img, (y1, x1), (y2, x2), (0, 255, 0), thickness=2)

	show(img, from_cv2=True)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', required=True, type=str)
	main(parser.parse_args())
