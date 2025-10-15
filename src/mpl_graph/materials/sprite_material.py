# pip imports
import numpy as np

# local imports
from .material import Material
from ..core import Constants, Texture


class SpriteMaterial(Material):
    """A simple sprite material class to hold sprite material properties."""

    __slots__ = "texture"

    def __init__(
        self,
        texture: Texture | None = None,
        extent: tuple[float, float, float, float] | None = None,
    ):
        super().__init__()

        self.texture: Texture = texture if texture is not None else Texture()
        """Texture for the sprite."""

        self.extent: tuple[float, float, float, float] = extent if extent is not None else (-0.5, 0.5, -0.5, 0.5)
        """Extent of the sprite in local space (left, right, bottom, top)."""
