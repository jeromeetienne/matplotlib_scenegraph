# stdlib imports
from typing import Literal

# pip imports
import numpy as np

# local imports
from .material import Material
from ..core import Constants, Texture


class TextMaterial(Material):
    """A simple text material class to hold text material properties."""

    __slots__ = "font_size", "color"

    def __init__(
        self,
        font_size: int = 12,
        color: np.ndarray | None = None,
        horizontal_align: Literal["left", "center", "right"] = "center",
        vertical_align: Literal["bottom", "baseline", "center", "center_baseline", "top"] = "center",
    ):
        super().__init__()

        self.font_size = font_size
        """Font size in points."""
        self.color = color if color is not None else Constants.Color.WHITE
        """Color as an numpy array."""
        self.horizontal_align: Literal["left", "center", "right"] = horizontal_align
        """Horizontal alignment."""
        self.vertical_align: Literal["bottom", "baseline", "center", "center_baseline", "top"] = vertical_align
        """Vertical alignment."""
