import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Create figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.2, 1.2)

# Initial artist (line or point)
(point,) = ax.plot([], [], "o")  # using a single point


# Initialization function (called once)
def init():
    point.set_data([], [])
    return [point]


# Animation function (called every frame)
def update(frame):
    x = frame
    y = np.sin(frame)
    point.set_data(x, y)
    return [point]  # must return a tuple


# Generate frames
frames = np.linspace(0, 2 * np.pi, 200)

anim = FuncAnimation(
    fig, update, frames=frames, init_func=init, blit=True, interval=20, repeat=True  # <-- HERE: enables blitting for speed  # milliseconds per frame
)

plt.show()
