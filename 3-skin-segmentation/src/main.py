import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import imutils


def _test():
	lower = np.array([0, 48, 80], dtype="uint8")
	upper = np.array([20, 255, 255], dtype="uint8")
	camera = cv2.VideoCapture(0)
	# keep looping over the frames in the video
	while True:
		# grab the current frame
		(grabbed, frame) = camera.read()

		# if we are viewing a video and we did not grab a
		# frame, then we have reached the end of the video
		if not grabbed:
			break

		# resize the frame, convert it to the HSV color space,
		# and determine the HSV pixel intensities that fall into
		# the speicifed upper and lower boundaries
		frame = imutils.resize(frame, width=400)
		converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		skinMask = cv2.inRange(converted, lower, upper)
		# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
		# skinMask = cv2.dilate(skinMask, kernel, iterations=1)
		# skinMask = cv2.erode(skinMask, kernel, iterations=1)
		# skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

		# show the skin in the image along with the mask
		cv2.imshow("images", np.hstack([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), skinMask]))

		# if the 'q' key is pressed, stop the loop
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break


def process_single(args):
	img = cv2.imread(args.img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

	mask = cv2.inRange(
		hsv, np.array(args.low_thr), np.array(args.high_thr)
	)

	assert mask[mask == 255].size + mask[mask == 0].size == mask.size

	strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

	# mask = cv2.medianBlur(mask, ksize=7)
	mask = cv2.erode(mask, strel, iterations=2)
	# mask = cv2.medianBlur(mask, ksize=11)
	# mask = cv2.erode(mask, strel, iterations=3)
	mask = cv2.dilate(mask, strel, iterations=3)
	# mask = cv2.erode(mask, strel, iterations=3)
	# mask = cv2.dilate(mask, strel, iterations=20)
	# mask = cv2.erode(mask, strel, iterations=5)
	# mask = cv2.medianBlur(mask, ksize=11)

	plt.imshow(np.vstack((gray, mask)), cmap='gray')
	# plt.imshow(mask, cmap='gray')
	plt.tight_layout()
	plt.show()


def main(args):
	process_single(args)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img_path', required=True, type=str)
	parser.add_argument('--low_thr', required=True, nargs='+', type=int)
	parser.add_argument('--high_thr', required=True, nargs='+', type=int)
	parser.add_argument('--debug', action='store_true')
	args = parser.parse_args()

	print(args)

	if not args.debug:
		main(args)
	else:
		_test()
