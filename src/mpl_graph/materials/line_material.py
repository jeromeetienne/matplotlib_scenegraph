# pip imports
import numpy as np

# local imports
from .material import Material
from ..core.constants import Constants


class LineMaterial(Material):
    """A simple line material class to hold line material properties."""

    def __init__(self, color: np.ndarray = Constants.Color.GRAY):
        super().__init__()
        self.color: np.ndarray = color
