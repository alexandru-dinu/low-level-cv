import matplotlib.pylab as plt
import numpy as np


def gaussian_kernel(sigma=1, size=3):
    # range [-k, k] from size
    k = size // 2
    ps = [np.exp(-(z * z / (2 * sigma ** 2))) for z in range(-k, k + 1)]
    f = np.outer(ps, ps)
    f = f / f.sum()
    return f


def box_kernel(size=3):
    f = np.ones((size, size)) / size ** 2
    return f


def sobel_kernels():
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    return sx, sy


def stop_sign_kernels():
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

    sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    sd1 = np.array([[0, -2, 0], [-2, 0, 2], [0, 2, 0]], dtype=np.float32)

    sd2 = np.array([[0, 2, 0], [2, 0, -2], [0, -2, 0]], dtype=np.float32)

    return sx, sy, sd1, sd2
