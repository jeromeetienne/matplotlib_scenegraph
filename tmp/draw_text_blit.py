import matplotlib.pyplot as plt

fig, ax = plt.subplots()
text = ax.text(0.5, 0.5, "Initial", ha="center", va="center")
plt.draw()  # initial draw to cache renderer

background = fig.canvas.copy_from_bbox(ax.bbox)

for new_text in ["One", "Two", "Three"]:
    fig.canvas.restore_region(background)
    text.set_text(new_text)
    ax.draw_artist(text)
    fig.canvas.blit(ax.bbox)
    # print the size of ax.bbox
    print(ax.bbox.width, ax.bbox.height)
    plt.pause(0.5)
