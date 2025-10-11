import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Example line segments: list of ((x0, y0), (x1, y1)), ...
segments = [[(0, 0), (1, 1)], [(1, 0), (0, 1)], [(0.5, 0), (0.5, 1)]]

# Create the LineCollection object, with optional styling
line_collection = LineCollection(segments, colors="blue", linewidths=2)

# Create a figure and axes
fig, ax = plt.subplots()

# Add the collection to the axes
ax.add_collection(line_collection)

# Set axes limits
ax.set_xlim(-1, 2)
ax.set_ylim(-1, 2)

# Display the plot
plt.show()
