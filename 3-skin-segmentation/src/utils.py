import json
import urllib.request as req
from random import shuffle
import argparse


def download_images(count=10):
	data = [json.loads(x.strip()) for x in open("../data/face_detection.json")]
	shuffle(data)

	for i in range(count):
		img = open(f"../data/img_{i}.jpg", 'wb')
		img.write(req.urlopen(data[i]['content']).read())
		img.close()

		print(f"Downloaded ../data/img_{i}.jpg")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num', required=True, type=int)
	args = parser.parse_args()

	download_images(args.num)
