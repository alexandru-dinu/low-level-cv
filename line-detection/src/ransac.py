import numpy as np
import random
from utils import show


def fit_lsq(x, y):
    """
    return linear model: a*x + b
    """

    x = np.matrix(np.vstack((x, np.ones(len(x))))).T
    y = np.matrix(y).T

    model_vec = np.linalg.inv(x.T * x) * x.T * y

    a = model_vec[0, 0]
    b = model_vec[1, 0]

    return model_vec, lambda p: a * p + b


def dist(model, x, y):
    return np.abs(y - model(x))


def ransac(data, sample_size=10, num_iter=100, inlier_thr=10, target_inliers=None):
    """
    """

    x, y = np.where(data == 1)
    z = list(zip(x, y))

    best_inlier_count = 0
    best_model = None
    best_inliers = None

    for it in range(num_iter):
        sample = random.sample(z, sample_size)

        sx = np.array([e[0] for e in sample])
        sy = np.array([e[1] for e in sample])

        model_vec, model = fit_lsq(sx, sy)
        # print(f"[{it}] -> {model_vec.T}")

        inliers = [(px, py) for (px, py) in zip(x, y) if dist(model, px, py) <= inlier_thr]

        if len(inliers) > best_inlier_count:
            best_inlier_count = len(inliers)
            # best_model = (model_vec, model)
            best_inliers = inliers[:]

            best_model = fit_lsq(
                x=np.array([e[0] for e in inliers]),
                y=np.array([e[1] for e in inliers])
            )

            if target_inliers is not None and len(inliers) > target_inliers:
                break

        if it % (num_iter // 10) == 0:
            print(f"[+] iter {it}")

    print(f"[+] Took {it} iterations. Best inlier count = {best_inlier_count}.")

    return best_model, best_inliers


def strip_inliers(data, inliers):
    print(f"[+] Removing {len(inliers)} inliers")
    for px, py in inliers:
        data[px, py] = 0
    return data


def find_sequential(data, config):
    cdata = data.copy()

    models = []
    model_vecs = []

    for line in range(config['num_lines']):
        (model_vec, model), inliers = ransac(cdata, config['sample_size'], config['num_iter'], config['inlier_thr'])

        cdata = strip_inliers(cdata, inliers)
        # show(cdata, cmap='gray')

        models.append(model)
        model_vecs.append(model_vec)

        print(f"[+] [Fitted line idx {line}] Best model: {model_vec.T}")

    return model_vecs, models
