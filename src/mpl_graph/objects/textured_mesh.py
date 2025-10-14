# pip imports
import numpy as np

# local imports
from ..core.object_3d import Object3D
from ..core.texture import Texture
from ..geometry import MeshGeometry
from ..materials import TextureMeshMaterial


class TexturedMesh(Object3D):
    __slots__ = ("geometry", "material")

    def __init__(self, geometry: MeshGeometry | None = None, material: TextureMeshMaterial | None = None) -> None:
        super().__init__()

        self.name = f"a {TexturedMesh.__name__}"
        self.geometry: MeshGeometry = geometry if geometry is not None else MeshGeometry()
        """Geometry of the textured mesh."""
        self.material: TextureMeshMaterial = material if material is not None else TextureMeshMaterial()
        """Material of the textured mesh."""

        self.sanity_checks()

    def sanity_checks(self) -> None:
        """Perform sanity checks on the geometry and material of the textured mesh."""
        assert self.geometry.uvs is not None, f"The geometry must have texture coordinates (uvs) defined for a textured mesh"
        assert self.material.texture is not None, f"The material must have a texture defined for a textured mesh"
        assert len(self.geometry.uvs) == len(
            self.geometry.vertices
        ), f"The number of uvs must be equal to the number of vertices, got {len(self.geometry.uvs)} uvs and {len(self.geometry.vertices)} vertices"
