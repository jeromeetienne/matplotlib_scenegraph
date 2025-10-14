# pip imports
import numpy as np

# local imports
from .material import Material
from ..core.constants import Constants


class LinesMaterial(Material):
    """A simple line material class to hold line material properties."""

    __slots__ = ("colors",)

    def __init__(self, colors: np.ndarray | None = None):
        super().__init__()

        self.colors: np.ndarray = colors if colors is not None else np.array([Constants.Color.GRAY])
