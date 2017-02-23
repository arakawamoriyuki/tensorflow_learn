import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-3, 3, 0.1)
y = np.sin(x)

metrics = np.random.rand(10, 10)

plt.plot(x, y)
plt.hist(x)
plt.imshow(metrics)

plt.show()