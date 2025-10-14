# pip imports
import numpy as np

from mpl_graph.core.constants import Constants

# local imports
from .material import Material
from ..core import Texture


class MeshPhongMaterial(Material):
    """A simple material class to hold texture mesh material properties."""

    __slots__ = "texture"

    def __init__(
        self,
        texture: Texture | None = None,
        face_sorting: bool | None = None,
        face_culling: Constants.FaceCulling | None = None,
    ):
        super().__init__()

        self.texture: Texture = texture if texture is not None else Texture()
        """Texture for the TextureMeshMaterial."""
        self.face_sorting: bool = face_sorting if face_sorting is not None else True
        """Whether to sort faces by depth (painter's algorithm)."""
        self.face_culling: Constants.FaceCulling = face_culling if face_culling is not None else Constants.FaceCulling.FrontSide
        """Whether to cull faces based on their orientation relative to the camera."""
