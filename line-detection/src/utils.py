import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_edges(img):
    out = img.copy()

    out = cv2.GaussianBlur(out, ksize=(3, 3), sigmaX=0)
    out = cv2.Canny(out, 170, 230)

    return out


def morph_close(img):
    out = img.copy()

    strel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(7, 7))

    out = cv2.dilate(out, strel, iterations=1)
    out = cv2.erode(out, strel, iterations=1)

    return out


def open_img(img_path, gray=False):
    if gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)

    return img


def show(arr, from_cv2=False, cmap=None):
    if from_cv2:
        b = arr[:, :, 0]
        g = arr[:, :, 1]
        r = arr[:, :, 2]
        arr = np.dstack((r, g, b))

    plt.imshow(arr, cmap=cmap)
    plt.tight_layout()

    plt.show()
