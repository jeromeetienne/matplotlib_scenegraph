import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# --- Step 1: Create a blank image (as background) ---
width, height = 800, 600
img = Image.new("RGB", (width, height), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

# --- Step 2: Optional: Load a TTF font for good quality (or default) ---
try:
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()

# --- Step 3: Draw A LOT of text FAST ---
for i in range(2000):
    x = np.random.randint(0, width)
    y = np.random.randint(0, height)
    draw.text((x, y), f"T{i}", font=font, fill=(0, 0, 0))

# --- Step 4: Convert to array and render with imshow ---
img_np = np.array(img)

plt.imshow(img_np)
plt.axis("off")
plt.show()
