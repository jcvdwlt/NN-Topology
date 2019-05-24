import numpy as np
from matplotlib import pyplot as plt

N_POINTS = 20000

x1 = np.random.random(int(N_POINTS / 2))
x2a = 2*(x1-0.5)**2 + 0.1*np.random.rand(len(x1))
x2b = 2*(x1-0.5)**2 + 0.25 + 0.1*np.random.rand(len(x1))

ya = np.ones(len(x1))
yb = np.zeros(len(x1))

data = np.vstack((np.concatenate((x1, x1)), np.concatenate((x2a, x2b)), np.concatenate((ya, yb)))).transpose()
np.random.shuffle(data)

np.save('features', data[:, 0:2])
np.save('targets', data[:, 2])

plt.scatter(data[0:500, 0], data[0:500, 1], c=data[0:500, 2])
plt.show()



