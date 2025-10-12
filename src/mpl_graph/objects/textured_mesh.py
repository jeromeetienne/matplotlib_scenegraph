# pip imports
import numpy as np

# local imports
from ..core.object_3d import Object3D
from ..core.texture import Texture


class TexturedMesh(Object3D):
    __slots__ = ("texture", "vertices", "indices", "uvs")

    def __init__(self, faces_indices: np.ndarray, vertices_coords: np.ndarray, uvs_coords: np.ndarray, texture: Texture) -> None:
        super().__init__()

        # sanity check - make sure we have triangular faces
        assert faces_indices.ndim == 2 and faces_indices.shape[1] == 3, "faces_indices should be of shape [M, 3]"
        assert vertices_coords.ndim == 2 and vertices_coords.shape[1] == 3, "vertices_coords should be of shape [N, 3]"
        assert uvs_coords.ndim == 2 and uvs_coords.shape[1] == 2, "uvs_coords should be of shape [N, 2]"
        assert len(vertices_coords) == len(uvs_coords), "The number of vertices must be equal to the number of texture coordinates"

        # copy the arguments
        # TODO to change in .triangle_count  and reshape vertices to (M, 3, 3)
        # - store .vertices and .uvs as (N, 3) and (N, 2)

        # remove the alpha channel if any - FIXME is that needed ?
        texture = texture.strip_alpha() if texture.has_alpha() else texture

        self.name = f"a {TexturedMesh.__name__}"
        self.vertices = vertices_coords
        self.indices = faces_indices
        self.uvs = uvs_coords
        self.texture = texture
