import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Define vertices of the polygon as a list of (x, y) tuples
vertices = [(1, 1), (2, 1), (2, 2), (1, 2), (0.5, 1.5)]

# Create a Polygon patch with the vertices, specify face color if desired
polygon = Polygon(vertices, closed=True, facecolor="blue", edgecolor="black")

fig, ax = plt.subplots()
ax.add_patch(polygon)

# Set the limits of the plot to fit the polygon
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

plt.show()
