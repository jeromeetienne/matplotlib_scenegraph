import matplotlib.pyplot as plt
import numpy as np

# Example: Draw 3 lines
x = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])  # Line 1: x-coordinates  # Line 2: x-coordinates  # Line 3: x-coordinates
y = np.array([[1, 2, 3], [2, 3, 4], [3, 2, 1]])  # Line 1: y-coordinates  # Line 2: y-coordinates  # Line 3: y-coordinates

plt.plot(x.T, y.T)  # Transpose if lines are rows
plt.show()
