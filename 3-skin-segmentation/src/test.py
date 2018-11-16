import numpy as np
import matplotlib.pyplot as plt

# Generate fake data.

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

ellipse = np.sqrt(2) * np.c_[s[0] * np.cos(2 * np.pi * t), s[1] * np.sin(2 * np.pi * t)]
angle = np.arctan2(vt[0, 1], vt[0, 0])
ellipse = np.dot(rotation_matrix(angle), ellipse.T).T

plt.plot(X[:, 0], X[:, 1], 'ob')
plt.plot(ellipse[:, 0], ellipse[:, 1], '-r', lw=3)
plt.axis('equal')
plt.grid()
plt.show()
