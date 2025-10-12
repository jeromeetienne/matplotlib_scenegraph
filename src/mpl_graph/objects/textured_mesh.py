# pip imports
import numpy as np

# local imports
from ..core.object_3d import Object3D
from ..core.texture import Texture
from ..core.geometry import Geometry


class TexturedMesh(Object3D):
    __slots__ = ("geometry", "texture")

    def __init__(self, geometry: Geometry, texture: Texture) -> None:
        super().__init__()

        # copy the arguments
        # TODO to change in .triangle_count  and reshape vertices to (M, 3, 3)
        # - store .vertices and .uvs as (N, 3) and (N, 2)

        # remove the alpha channel if any - FIXME is that needed ?
        texture = texture.strip_alpha() if texture.has_alpha() else texture

        self.name = f"a {TexturedMesh.__name__}"
        self.geometry = geometry
        self.texture = texture
