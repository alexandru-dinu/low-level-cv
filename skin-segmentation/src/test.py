import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2
from utils import get_ellipse

np.set_printoptions(suppress=True, precision=4)

rs = np.random.RandomState(12345)

h, w = 500, 500
img = np.zeros((h, w), dtype=np.uint8)

a = 50
b = 150
N = 500
t = np.linspace(0, 1, N, endpoint=True)

rotation_matrix = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])

X = np.vstack((
	np.c_[a * np.cos(2 * np.pi * t), b * np.sin(2 * np.pi * t)],
	np.c_[48 * np.cos(2 * np.pi * t), 147 * np.sin(2 * np.pi * t)],
	np.c_[38 * np.cos(2 * np.pi * t), 137 * np.sin(2 * np.pi * t)],
	np.c_[28 * np.cos(2 * np.pi * t), 127 * np.sin(2 * np.pi * t)],
))

N *= 4

X = X + rs.randn(N, 2) * 2.7
X = np.dot(rotation_matrix(np.deg2rad(37.5)), X.T).T
X = np.round(X + np.array([-X[:, 0].min(), -X[:, 1].min()]))

assert X.min() >= 0 and X.max() >= 0

xs = []

for px, py in X:
	px, py = int(px), int(py)
	xs.append((px+150, py+123))
xs = np.array(xs)

for px, py in xs:
	img[px, py] = 255

ellipse = get_ellipse(xs)

ellipse2 = cv2.fitEllipse(xs)
cv2.ellipse(img, ellipse2, 255, 2)

plt.imshow(img, cmap='gray')
plt.show()
exit(0)

exit(0)
# # Fit the ellipse.
# u, s, vt = np.linalg.svd((X - X.mean(axis=0)) / np.sqrt(N), full_matrices=False)
# print(s)


# v1 = (v[0,0], v[0,1])
# v2 = (v[1,0], v[1,1])

# ax = plt.axes()
# ax.arrow(0, 0, *v1, head_width=0.1, head_length=0.1)
# ax.arrow(0, 0, *v2, head_width=0.1, head_length=0.1)

# print(vt)
# print(v)

# ellipse = np.sqrt(2) * np.c_[s[0] * np.cos(2 * np.pi * t), s[1] * np.sin(2 * np.pi * t)]
# angle = np.arctan2(vt[0, 0], vt[1,0])
# ellipse = np.dot(rotation_matrix(angle), ellipse.T).T
# print(ellipse.shape)
