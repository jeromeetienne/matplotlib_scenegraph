# pip imports
import numpy as np

# local imports
from .light import Light
from ..core.constants import Constants


class AmbientLight(Light):
    def __init__(self, color: np.ndarray | None = None, intensity: float | None = None):
        super().__init__()

        self.name = f"a {AmbientLight.__name__}"

        self.color = color if color is not None else Constants.Color.WHITE
        """Base color of the light, as an (R, G, B) array with values in [0, 1]. shape (3,)"""
        self.intensity = intensity if intensity is not None else 1.0
        """Intensity of the light, as a float multiplier. Default is 1.0"""
