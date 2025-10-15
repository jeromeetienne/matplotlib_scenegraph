import matplotlib.pyplot as plt

fig, ax = plt.subplots()
text = ax.text(0.5, 0.5, "Initial", ha="center", va="center")
plt.draw()  # initial draw to cache renderer

fig.canvas.copy_from_bbox = fig.canvas.copy_from_bbox  # type: ignore

background = fig.canvas.copy_from_bbox(ax.bbox)

for text_index, new_text in enumerate(["One", "Two", "Three"]):
    fig.canvas.restore_region(background)
    text.set_text(new_text)
    ax.draw_artist(text)
    fig.canvas.blit(ax.bbox)
    # print the size of ax.bbox
    print(ax.bbox.width, ax.bbox.height)
    # change the position of the text
    text.set_position((0.5, 0.5 + 0.1 * text_index))
    plt.pause(0.5)
