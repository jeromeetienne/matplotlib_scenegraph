# stdlib imports
import typing
from typing import Sequence

# pip imports
import matplotlib.artist
import matplotlib.image
import matplotlib.path
import matplotlib.axes
import matplotlib.transforms
import matplotlib.collections
import numpy as np

from mpl_graph.core.constants import Constants
from mpl_graph.core.texture import Texture

# local imports
from ..objects.mesh import Mesh
from .renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils
from ..materials import MeshPhongMaterial
from .renderer_mesh import RendererMesh


class RendererMeshPhongMaterialNoTexture:
    # =============================================================================
    #
    # =============================================================================
    @staticmethod
    def render(
        renderer: Renderer,
        mesh: Mesh,
        camera: CameraBase,
        faces_vertices_world: np.ndarray,
        faces_vertices_2d: np.ndarray,
        light_intensities: np.ndarray,
        faces_visible: np.ndarray,
    ) -> list[matplotlib.artist.Artist]:

        material = typing.cast(MeshPhongMaterial, mesh.material)
        assert material.texture is None or material.texture.data.size == 0, "MeshPhongMaterialNoTexture should be used with no texture"

        # remove hidden faces
        faces_vertices_2d = faces_vertices_2d[faces_visible]
        light_intensities = light_intensities[faces_visible]

        faces_color = np.zeros((len(faces_vertices_2d), 4), dtype=np.float32)
        faces_color[:, 0] = light_intensities  # R
        faces_color[:, 1] = light_intensities  # G
        faces_color[:, 2] = light_intensities  # B
        faces_color[:, 3] = 1.0  # A

        # =============================================================================
        # Create artists if needed
        # =============================================================================
        if mesh.uuid not in renderer._artists:
            mpl_poly_collection = matplotlib.collections.PolyCollection([], clip_on=False, snap=False)
            mpl_poly_collection.set_visible(False)  # hide until properly positioned and sized
            renderer._axis.add_collection(mpl_poly_collection)
            renderer._artists[mesh.uuid] = mpl_poly_collection

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_poly_collection = typing.cast(matplotlib.collections.PolyCollection, renderer._artists[mesh.uuid])
        mpl_poly_collection.set_visible(True)

        # =============================================================================
        # Update all the artists
        # =============================================================================

        # update the PathCollection with the new patches
        mpl_poly_collection.set_verts(typing.cast(list, faces_vertices_2d))
        mpl_poly_collection.set_facecolor(typing.cast(list, faces_color))
        mpl_poly_collection.set_edgecolor((0, 0, 0, 0.3))
        mpl_poly_collection.set_linewidth(0.5)

        return [mpl_poly_collection]
