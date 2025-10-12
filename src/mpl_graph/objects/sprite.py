# pip imports
from pyrr import vector3
import numpy as np

# Local imports
from ..core.constants import Constants
from ..core.object_3d import Object3D
from ..core.texture import Texture


class Sprite(Object3D):
    __slots__ = "texture"

    def __init__(self, texture: Texture) -> None:
        super().__init__()

        self.name = f"a {Sprite.__name__}"
        self.texture: Texture = texture
