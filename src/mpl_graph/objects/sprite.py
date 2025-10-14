# pip imports
from pyrr import vector3
import numpy as np

# Local imports
from ..core.constants import Constants
from ..core.object_3d import Object3D
from ..core.texture import Texture
from ..materials import SpriteMaterial


class Sprite(Object3D):
    __slots__ = "material"

    def __init__(self, material: SpriteMaterial | None = None) -> None:
        super().__init__()

        self.name = f"a {Sprite.__name__}"
        self.material: SpriteMaterial = material if material is not None else SpriteMaterial()
        """Material of the sprite."""
