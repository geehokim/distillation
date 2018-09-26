import matplotlib.pyplot as plt
import numpy as np
import torch

x = []
for i in range(4):
    x.append(i+1)

plt.plot(x, [10, 20, 25, 30], color='lightblue', linewidth=3)
plt.plot(x, [1, 2, 2, 3], color='darkgreen', linewidth=3)

plt.xlim(0.5, 4.5)
plt.ylim(0, 100)
plt.show()