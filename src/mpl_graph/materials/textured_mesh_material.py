# pip imports
import numpy as np

# local imports
from .material import Material
from ..core import Texture


class TextureMeshMaterial(Material):
    """A simple material class to hold texture mesh material properties."""

    __slots__ = "texture"

    def __init__(
        self,
        texture: Texture | None = None,
    ):
        super().__init__()

        self.texture: Texture = texture if texture is not None else Texture()
        """Texture for the TextureMeshMaterial."""
