# pip imports
import numpy as np

# local imports
from ..core.object_3d import Object3D
from ..core.texture import Texture
from ..geometry.geometry import Geometry


class TexturedMesh(Object3D):
    __slots__ = ("geometry", "texture")

    def __init__(self, geometry: Geometry, texture: Texture) -> None:
        super().__init__()

        self.name = f"a {TexturedMesh.__name__}"
        self.geometry = geometry
        self.texture = texture
