import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2

np.set_printoptions(suppress=True, precision=4)

rs = np.random.RandomState(12345)

a = 2
b = 7
N = 100
t = np.linspace(0, 1, N, endpoint=False)

rotation_matrix = lambda x: np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])

X = np.c_[a * np.cos(2 * np.pi * t), b * np.sin(2 * np.pi * t)]
X = X + rs.randn(N, 2) * 0.7
X = np.dot(rotation_matrix(np.pi/6), X.T).T

# Fit the ellipse.
u, s, vt = np.linalg.svd((X - X.mean(axis=0)) / np.sqrt(N), full_matrices=False)
print(s)

x = X
xp = np.matrix(x - x.mean(axis=0))
cm = (1 / (len(xp) - 1)) * xp.T * xp
l, v = np.linalg.eig(cm)
mv = v + x.mean(axis=0)

v1 = (v[0,0], v[0,1])
v2 = (v[1,0], v[1,1])

ax = plt.axes()
ax.arrow(0, 0, *v1, head_width=0.1, head_length=0.1)
ax.arrow(0, 0, *v2, head_width=0.1, head_length=0.1)

print(vt)
print(v)

ellipse = np.sqrt(2) * np.c_[s[0] * np.cos(2 * np.pi * t), s[1] * np.sin(2 * np.pi * t)]
angle = np.arctan2(vt[0, 0], vt[1,0])
ellipse = np.dot(rotation_matrix(angle), ellipse.T).T
print(ellipse.shape)

plt.plot(x[:, 0], x[:, 1], 'ob')
plt.plot(ellipse[:, 0], ellipse[:, 1], '-r', lw=3)
plt.ylim(-10,10)
plt.xlim(-10,10)
plt.tight_layout()
plt.grid()
plt.show()
