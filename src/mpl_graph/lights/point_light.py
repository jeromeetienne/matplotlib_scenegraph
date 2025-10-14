# pip imports
import numpy as np

# local imports
from .light import Light
from ..core.constants import Constants


class PointLight(Light):
    def __init__(self, color: np.ndarray | None = None, intensity: float | None = None):
        super().__init__()

        self.name = f"a {PointLight.__name__}"
        self.color = color if color is not None else Constants.Color.WHITE
        self.intensity = intensity if intensity is not None else 1.0
