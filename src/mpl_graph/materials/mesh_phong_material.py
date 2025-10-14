# pip imports
import numpy as np

from mpl_graph.core.constants import Constants

# local imports
from .material import Material
from ..core import Texture, Constants
from .mesh_material import MeshMaterial


class MeshPhongMaterial(MeshMaterial):
    """A simple material class to hold texture mesh material properties."""

    __slots__ = "texture"

    def __init__(
        self,
        color: np.ndarray | None = None,
        shininess: float | None = None,
        face_sorting: bool | None = None,
        face_culling: Constants.FaceCulling | None = None,
        edge_colors: np.ndarray | None = None,
        edge_widths: np.ndarray | None = None,
    ):
        super().__init__()

        self.color: np.ndarray = color if color is not None else Constants.Color.CYAN
        """Base color of the material, as an (R, G, B) array with values in [0, 1]. shape (3,)"""
        self.shininess: float = shininess if shininess is not None else 30.0
        """Shininess factor for specular highlights."""
        self.face_sorting: bool = face_sorting if face_sorting is not None else True
        """Whether to sort faces by depth (painter's algorithm)."""
        self.face_culling: Constants.FaceCulling = face_culling if face_culling is not None else Constants.FaceCulling.FrontSide
        """Whether to cull faces based on their orientation relative to the camera."""
        self.edge_colors: np.ndarray = edge_colors if edge_colors is not None else np.array([Constants.Color.BLACK])
        """array of point edge colors, shape (N, 3) or (N, 4)"""
        self.edge_widths: np.ndarray = edge_widths if edge_widths is not None else np.array([0.1])
        """array of point edge widths, shape (N,)"""
