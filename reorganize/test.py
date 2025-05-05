import numpy as np
import matplotlib.pyplot as plt


list = [
    np.array([0, 0, 0, 0, 0]),
    np.array([0, 1, 0, 1, 0]),
    np.array([0, 0, 0, 0, 0]),
    np.array([0, 1, 1, 1, 0]),
    np.array([0, 0, 0, 0, 0]),
]

data = np.empty((0, 0))

for i, val in enumerate(list):
    data[i, :] = list[i]

plt.imshow(data, cmap="hot", interpolation="nearest")
plt.colorbar()
plt.show()
