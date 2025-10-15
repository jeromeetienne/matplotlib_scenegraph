# pip imports
import numpy as np

# local imports
from .material import Material
from ..core.constants import Constants


class PointsMaterial(Material):
    """A simple line material class to hold line material properties."""

    __slots__ = ("colors", "sizes", "edge_colors", "edge_widths")

    def __init__(
        self,
        colors: np.ndarray | None = None,
        sizes: np.ndarray | None = None,
        edge_colors: np.ndarray | None = None,
        edge_widths: np.ndarray | None = None,
        depth_sorting: bool | None = None,
    ):
        super().__init__()

        self.colors: np.ndarray = colors if colors is not None else np.array([Constants.Color.GRAY])
        """array of point colors, shape (N, 3) or (N, 4)"""
        self.sizes: np.ndarray = sizes if sizes is not None else np.array([10.0])
        """array of point sizes, shape (N,)"""
        self.edge_colors: np.ndarray = edge_colors if edge_colors is not None else np.array([Constants.Color.BLACK])
        """array of point edge colors, shape (N, 3) or (N, 4)"""
        self.edge_widths: np.ndarray = edge_widths if edge_widths is not None else np.array([1.0])
        """array of point edge widths, shape (N,)"""
        self.depth_sorting: bool = depth_sorting if depth_sorting is not None else True
        """Whether to enable depth sorting based on camera distance at the point level."""
