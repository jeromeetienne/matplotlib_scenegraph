# pip imports
import numpy as np
from pyrr import vector3, vector4


class Constants:
    WHITE = vector4.create(1.0, 1.0, 1.0, 1.0)
    BLACK = vector4.create(0.0, 0.0, 0.0, 1.0)
    RED = vector4.create(1.0, 0.0, 0.0, 1.0)
    GREEN = vector4.create(0.0, 1.0, 0.0, 1.0)
    BLUE = vector4.create(0.0, 0.0, 1.0, 1.0)
    YELLOW = vector4.create(1.0, 1.0, 0.0, 1.0)
    CYAN = vector4.create(0.0, 1.0, 1.0, 1.0)
    MAGENTA = vector4.create(1.0, 0.0, 1.0, 1.0)
    GRAY = vector4.create(0.5, 0.5, 0.5, 1.0)
    LIGHT_GRAY = vector4.create(0.75, 0.75, 0.75, 1.0)
    DARK_GRAY = vector4.create(0.25, 0.25, 0.25, 1.0)
    ORANGE = vector4.create(1.0, 0.65, 0.0, 1.0)
    PURPLE = vector4.create(0.5, 0.0, 0.5, 1.0)
