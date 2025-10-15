# pip imports
import numpy as np

# local imports
from .material import Material
from ..core.constants import Constants


class PolygonsMaterial(Material):
    """A simple line material class to hold Polygons material properties."""

    __slots__ = ("colors", "sizes", "edge_colors", "edge_widths")

    def __init__(
        self,
        colors: np.ndarray | None = None,
        face_sorting: bool | None = None,
        face_culling: Constants.FaceCulling | None = None,
        sizes: np.ndarray | None = None,
        edge_colors: np.ndarray | None = None,
        edge_widths: np.ndarray | None = None,
    ):
        super().__init__()

        self.colors: np.ndarray = colors if colors is not None else np.array([Constants.Color.GRAY])
        """array of point colors, shape (N, 3) or (N, 4)"""
        self.depth_sorting: bool = face_sorting if face_sorting is not None else False
        """Whether to sort faces by depth (painter's algorithm)."""
        self.face_culling: Constants.FaceCulling = face_culling if face_culling is not None else Constants.FaceCulling.BothSides
        """Whether to cull faces based on their orientation relative to the camera."""
        self.sizes: np.ndarray = sizes if sizes is not None else np.array([10.0])
        """array of point sizes, shape (N,)"""
        self.edge_colors: np.ndarray = edge_colors if edge_colors is not None else np.array([Constants.Color.BLACK])
        """array of point edge colors, shape (N, 3) or (N, 4)"""
        self.edge_widths: np.ndarray = edge_widths if edge_widths is not None else np.array([1.0])
        """array of point edge widths, shape (N,)"""
