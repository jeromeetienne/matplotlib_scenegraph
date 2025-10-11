import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.collections
import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])

artist: matplotlib.collections.PathCollection = plt.scatter(x, y)  # 'b' for blue, 'o' for circle marker
plt.title("Plot of Points")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# artist.set_sizes


# Animation function for matplotlib
def update(frame):
    new_y = np.array([2, 3, 5, 7, 11]) + frame
    artist.set_offsets(np.c_[x, new_y])
    return (artist,)


ani = matplotlib.animation.FuncAnimation(plt.gcf(), update, frames=10, interval=500, blit=True)


plt.show()
