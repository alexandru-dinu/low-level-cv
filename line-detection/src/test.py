import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from termcolor import colored

from ransac import ransac, find_sequential

np.set_printoptions(suppress=True)


def generate(y, num_inliers=200, num_outliers=25):
    h, w = 300, 700
    xs = np.zeros((h, w), dtype=np.uint8)

    ns = np.linspace(1, num_inliers, num=num_inliers, dtype=np.uint8)

    ox = 33
    oy = 20

    for n in ns:
        rx = np.random.randint(20, dtype=np.uint8)
        ry = np.random.randint(30, dtype=np.uint8)
        xs[n + ox + rx, y(n) + oy + ry] = 1

    for _ in range(num_outliers):
        cx, cy = ns[ns.size // 2], y(ns[ns.size // 2])
        rx = np.random.randint(0, 2 * cx)
        ry = np.random.randint(0, 2 * cy)
        xs[rx, ry] = 1

    return xs


if __name__ == "__main__":
    data1 = generate(y=lambda x: 1 * x + 20, num_inliers=100, num_outliers=50)
    data2 = generate(y=lambda x: 2 * x - 33, num_inliers=100, num_outliers=50)
    data3 = generate(y=lambda x: 3 * x + 43, num_inliers=100, num_outliers=50)
    data4 = generate(y=lambda x: 4 * x - 20, num_inliers=100, num_outliers=50)

    data = np.hstack((data1, data2, data3, data4))

    # (model_vec, model), _ = ransac(data, sample_size=20, num_iter=2000, inlier_thr=10)
    # print(colored(f"Best model: {model_vec.T}", "green"))

    config = {"num_lines": 4, "sample_size": 5, "num_iter": 2000, "inlier_thr": 20}
    model_vecs, models = find_sequential(data, config)

    img = data.copy()

    for m in models:
        x1, x2 = 0, 200
        y1, y2 = int(m(x1)), int(m(x2))
        cv2.line(img, (y1, x1), (y2, x2), 1, thickness=1)

    plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show()

    pass
