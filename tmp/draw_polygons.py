import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Create a list of polygons, each polygon is a list of (x,y) points
polygons_coords = [[(1, 1), (2, 1), (2, 2), (1, 2)], [(3, 1), (4, 1), (4, 2), (3, 2)], [(1, 3), (2, 3), (2, 4), (1, 4)]]

# Create a polygon patch for each set of coordinates
patches = [Polygon(coords, closed=True) for coords in polygons_coords]

# Create a PatchCollection from the polygon patches
p = PatchCollection(patches, facecolor="orange", edgecolor="black", alpha=0.5)

fig, ax = plt.subplots()
ax.add_collection(p)

ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_aspect("equal")

plt.show()
