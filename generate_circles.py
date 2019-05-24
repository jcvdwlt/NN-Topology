import numpy as np
from matplotlib import pyplot as plt

N_POINTS = 20000

rad1 = 0.3 + 0.1 * np.random.rand(int(N_POINTS / 2))
rad2 = 0.25 * np.random.rand(int(N_POINTS / 2))
rad = np.concatenate((rad1, rad2))

angle = np.random.rand(N_POINTS)*np.pi*2
x = rad * np.cos(angle) + 0.5
y = rad * np.sin(angle) + 0.5

ya = np.ones(int(N_POINTS / 2))
yb = np.zeros(int(N_POINTS / 2))

data = np.vstack((x, y, np.concatenate((ya, yb)))).transpose()
np.random.shuffle(data)

np.save('features', data[:, 0:2])
np.save('targets', data[:, 2])

plt.scatter(data[0:500, 0], data[0:500, 1], c=data[0:500, 2])
plt.show()


