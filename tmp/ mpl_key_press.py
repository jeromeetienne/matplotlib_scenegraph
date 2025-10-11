import matplotlib.pyplot as plt
import numpy as np

def on_press(event):
    print('Pressed key:', event.key)

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)

# Plot something (optional)
ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)))
ax.set_title('Press any key')

plt.show()
qq