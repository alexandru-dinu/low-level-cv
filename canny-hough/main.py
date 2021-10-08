import time
from operator import itemgetter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.measure

import sys

SOBEL_GX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int32)

SOBEL_GY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.int32)


class Orientation:
    E_W = 0
    NE_SW = 1
    N_S = 2
    NW_SE = 3


CANNY_STRONG = 255
CANNY_WEAK = 100


def show_image(image, size, cmap="gray"):
    if size == "huge":
        figsize = (40, 30)
    elif size == "big":
        figsize = (30, 20)
    else:
        figsize = (20, 10)

    plt.figure(figsize=figsize)
    plt.imshow(image, cmap)
    plt.show()


def to_grayscale(image, coeffs=None):
    if coeffs is None:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        c = np.array(coeffs).reshape((1, 3))
        return cv2.transform(image, c)


def normalize(grad):
    grad = np.abs(grad * 255 / np.max(grad))
    grad = grad.astype("uint8")

    return grad


def compute_orientation(angles):
    """
    Given the set of angles from sobel, map each angle
    to the orientation it gives:

    [0,22.5) U [157.5, 180) -> east to west (horizontal gradient)
    [22.5, 67.5) -> north-east to south-west
    [67.5, 112.5) -> north to south (vertical gradient)
    [112.5, 157.5) -> north-west to south-east
    """
    height, width = angles.shape[:2]

    orientation = np.zeros_like(angles)

    for i in range(height):
        for j in range(width):
            d = np.rad2deg(angles[i, j]) % 180

            if (0 <= d < 22.5) or (157.5 <= d < 180):
                orientation[i, j] = Orientation.E_W

            elif 22.5 <= d < 67.5:
                orientation[i, j] = Orientation.NE_SW

            elif 67.5 <= d < 112.5:
                orientation[i, j] = Orientation.N_S

            elif 112.5 <= d < 157.5:
                orientation[i, j] = Orientation.NW_SE

    return orientation


def sobel(image):
    """
    Apply Gx and Gy Sobel convolutions to the grayscale image.
    Returns the normalized gradient and angles.
    """
    height, width = image.shape[:2]

    gx = np.zeros_like(image, dtype="float32")
    gy = np.zeros_like(image, dtype="float32")

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            patch = image[y - 1 : y + 2, x - 1 : x + 2]

            gx[y - 1, x - 1] = np.sum(patch * SOBEL_GX)
            gy[y - 1, x - 1] = np.sum(patch * SOBEL_GY)

    grad = np.hypot(gx, gy)
    angles = np.arctan2(gy, gx)

    return grad, gx, gy, angles


def compute_suppressed_gradient(grad, orient):
    """
    Look for local maxima w.r.t. gradient's orientation.
    Keep only the local maxima in the resulting suppressed gradient.
    """

    height, width = grad.shape[:2]

    suppressed_grad = np.zeros_like(grad)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # current orientation
            o = orient[y, x]

            # current gradient value
            g = grad[y, x]

            # horizontal gradient => check E/W
            if o == Orientation.E_W:
                if g >= grad[y, x - 1] and g >= grad[y, x + 1]:
                    suppressed_grad[y - 1, x - 1] = g

            # "second diagonal" gradient
            elif o == Orientation.NE_SW:
                if g >= grad[y - 1, x + 1] and g >= grad[y + 1, x - 1]:
                    suppressed_grad[y - 1, x - 1] = g

            # vertical gradient => check N/S
            elif o == Orientation.N_S:
                if g >= grad[y - 1, x] and g >= grad[y + 1, x]:
                    suppressed_grad[y - 1, x - 1] = g

            # "first diagonal" gradient
            elif o == Orientation.NW_SE:
                if g >= grad[y - 1, x - 1] and g >= grad[y + 1, x + 1]:
                    suppressed_grad[y - 1, x - 1] = g

    return suppressed_grad


def thresholding(image, low_threshold, high_threshold):
    # automatically accept (strong pixels)
    s_i, s_j = np.where(image > high_threshold)
    # accept in-between thresholds (weak pixels)
    w_i, w_j = np.where((image >= low_threshold) & (image <= high_threshold))
    # automatically reject
    r_i, r_j = np.where(image < low_threshold)

    # strong pixels
    image[s_i, s_j] = CANNY_STRONG
    # weak pixels
    image[w_i, w_j] = CANNY_WEAK
    # rejected pixels
    image[r_i, r_j] = 0

    return image


def edge_tracking(image):
    """
    Mark weak pixels as strong if they are connected to a strong pixel
    """
    height, width = image.shape[:2]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pixel = image[y][x]

            neighbours = [
                image[y - 1][x],
                image[y + 1][x],
                image[y][x - 1],
                image[y][x + 1],
                image[y - 1][x - 1],
                image[y + 1][x + 1],
                image[y + 1][x - 1],
                image[y - 1][x + 1],
            ]

            strong_neighbours = map(lambda p: p == CANNY_STRONG, neighbours)

            # if there is a weak pixel connected to a strong pixel,
            # mark it as strong
            if pixel == CANNY_WEAK:
                if any(strong_neighbours):
                    image[y][x] = CANNY_STRONG
                else:
                    image[y][x] = 0

    return image


# The gradient is always perpendicular to the edge.
# Intensities change across the edge
def canny(image, low_threshold, high_threshold):
    image = np.array(image, dtype=float)

    # get the gradients and the angles after applying Sobel convolutions
    gradients, gx, gy, angles = sobel(image)
    # gradients = normalize(gradients)

    # map each angle to its specific orientation
    orientation = compute_orientation(angles)

    # apply non-maximum suppression => thin-edges
    suppressed_grad = compute_suppressed_gradient(gradients, orientation)

    # apply thresholding (accept / reject pixels)
    leveled_img = thresholding(suppressed_grad, low_threshold, high_threshold)

    # adjust pixel strength w.r.t its neighbours
    final_image = edge_tracking(leveled_img)

    final_image = final_image.astype("uint8")

    return final_image


def hough(image, min_radius, max_radius, dist_between, count, optimize=False):
    height, width = image.shape[:2]

    radii = (max_radius - min_radius) + 1

    # stage 1: construct the accumulator matrix (confidence levels)

    # accumulator matrix
    A = np.zeros((radii, height, width))

    for y in range(max_radius, height - max_radius):
        for x in range(max_radius, width - max_radius):
            if image[y][x] == 0:
                continue

            if optimize:
                arr_r = np.arange(min_radius, max_radius + 1)
                arr_t = np.deg2rad(np.arange(361))

                arr_a = np.round(y - arr_r[:, None] * np.sin(arr_t)).astype("int")
                arr_b = np.round(x - arr_r[:, None] * np.cos(arr_t)).astype("int")

                # array containing indices that need to be incremented in A
                # [:, None] gives the transpose
                # A[r, a, b] = idx[r * h * w + a * w + b]
                idx = (
                    ((arr_r - min_radius) * height * width)[:, None]
                    + (arr_a * width)
                    + (arr_b)
                )

                # flatten and sort this array
                idx.ravel().sort()
                idx.shape = -1

                # get the start of each group of identical indices
                group_idx = np.flatnonzero(
                    np.concatenate(([True], idx[1:] != idx[:-1], [True]))
                )

                # omit the last one (it gives one after the end: len + 1)
                start_indices = group_idx[:-1]

                # increment x number of times, where each x is given by np.diff
                # x represents the number of occurrences of a given index (that needs to be incremented)
                A.flat[idx[start_indices]] += np.diff(group_idx)

            else:
                for r in range(min_radius, max_radius + 1):
                    for t in range(361):
                        trad = np.deg2rad(t)

                        b = x - r * np.cos(trad)
                        a = y - r * np.sin(trad)

                        b = np.floor(b).astype("int")
                        a = np.floor(a).astype("int")

                        A[r - min_radius, a, b] += 1

    # stage 2: find top `count` circles
    euclid = lambda p1, p2: np.linalg.norm(np.array(p1) - np.array(p2))

    A_sorted = np.sort(np.ravel(A))[::-1]

    circles = []

    for score in A_sorted:
        if len(circles) == count:
            break

        circle = np.argwhere(A == score)[0]

        curr_center = circle[1:]

        last_centers = [c[1:] for c in circles]
        far_enough = map(lambda l: euclid(l, curr_center) >= dist_between, last_centers)

        if all(far_enough):
            circles.append(circle)

    # fix radius size
    for c in circles:
        c[0] += min_radius

    return circles


def color_image(shape, gradients):
    # output: the colored image
    colored_image = np.zeros(shape, dtype=np.uint8)
    # 2D array that store the index of each gray pixel (the region it belongs to)
    area_ids = np.zeros(shape[:2], dtype=np.int32)

    # gradient is 0 if there's no edge in the image (background / inner region)
    # we need to separate this from 0
    # i.e. 255 becomes the index of the background
    area_ids[gradients == 0] = 255

    # label regions (i.e. assign an index for each element of the image)
    area_ids = skimage.measure.label(area_ids)

    # find the values that need to be filled
    filled_regions = scipy.ndimage.maximum_filter(area_ids, size=5)

    # "fill in the gaps"
    # 0 means unfilled
    inner_regions = area_ids == 0
    area_ids[inner_regions] = filled_regions[inner_regions]

    # we can now look far
    area_id, area_size = np.unique(area_ids, return_counts=True)
    areas_sorted = sorted(zip(area_id, area_size), key=itemgetter(1), reverse=True)

    # color the background with white
    colored_image[area_ids == areas_sorted[0][0]] = (255, 255, 255)

    # color the tree with green
    colored_image[area_ids == areas_sorted[1][0]] = (80, 176, 0)

    # color the trunk with brown
    colored_image[area_ids == areas_sorted[2][0]] = (0, 96, 127)

    # color the rest of the regions with yellow
    for area_id, _ in areas_sorted[3:]:
        colored_image[area_ids == area_id] = (102, 217, 240)

    return colored_image


def draw_circles(image, circles, conversion=False, filled=False, draw_center=False):
    cimg = None

    if conversion:
        cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        cimg = image.copy()

    thickness = -1 if filled else 2

    for c in circles:
        center = tuple(c[1:])[::-1]
        radius = c[0]
        cv2.circle(cimg, center, radius, (0, 0, 255), thickness)

        if draw_center:
            cv2.circle(cimg, center, 2, (0, 0, 255), 3)

    return cimg


def read_image(file_name, mode="BGR"):
    # read image as a numpy array
    img = cv2.imread(file_name)

    return img if mode == "BGR" else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    image_path = sys.argv[1]

    image = read_image(image_path)

    # brad1_bgr = [0.7, 0.25, 0.05]
    # brad1_rgb = [0.05, 0.25, 0.7]
    # brad2_bgr = [0.1, 0.55, 0.35]
    # brad2_rgb = [0.35, 0.55, 0.1]

    # coeffs for converting to grayscale should be deduced
    # from image's histogram

    t1 = time.perf_counter()
    gray_image = to_grayscale(image, [0.7, 0.25, 0.05])
    t2 = time.perf_counter()

    print("Grayscale conversion done. Took {0} s. ".format(np.round(t2 - t1, 5)))

    t1 = time.perf_counter()
    canny_image = canny(gray_image, 10, 20)
    t2 = time.perf_counter()

    print("Canny edge detection done. Took {0} s. ".format(np.round(t2 - t1, 5)))

    t1 = time.perf_counter()
    circles = hough(
        canny_image,
        min_radius=40,
        max_radius=60,
        dist_between=50,
        count=3,
        optimize=True,
    )
    t2 = time.perf_counter()

    for c in circles:
        print("\tRadius:", c[0], "Center:", c[1:])

    print("Hough circle detection done. Took {0} s. ".format(np.round(t2 - t1, 5)))

    t1 = time.perf_counter()
    _, gx, gy, _ = sobel(gray_image)
    gradients = np.hypot(normalize(gx), normalize(gy))

    recolored_image = color_image(image.shape, gradients)
    t2 = time.perf_counter()

    print("Recoloring done. Took {0} s. ".format(np.round(t2 - t1, 5)))

    final_image = draw_circles(recolored_image, circles, filled=True)
    canny_hough = draw_circles(canny_image, circles, conversion=True, draw_center=True)

    print("Everything done. Displaying images...")

    # # DEBUG
    # g = to_grayscale(recolored_image, [0.7, 0.25, 0.05])
    # ci = canny(g, 10, 20)
    # c = hough(ci, 40, 60, 50, 3, True)
    # f = draw_circles(ci, c, True, False, True)

    # cv2.imshow("TEST", f)
    # # DEBUG

    # show images
    cv2.imshow("Original image", image)
    cv2.imshow("Canny edges", canny_image)
    cv2.imshow("Hough circle detection", canny_hough)
    cv2.imshow("Recolored image", recolored_image)
    cv2.imshow("Final image", final_image)
    cv2.waitKey(0)
