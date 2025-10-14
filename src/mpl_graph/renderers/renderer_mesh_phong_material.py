# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.image
import matplotlib.path
import matplotlib.axes
import matplotlib.transforms
import numpy as np

from mpl_graph.core.constants import Constants
from mpl_graph.core.texture import Texture

# local imports
from ..objects.mesh import Mesh
from .renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils


class RendererMeshPhongMaterial:

    # =============================================================================
    # Update the artist for a texture face
    # =============================================================================
    @staticmethod
    def update_textured_face(
        mpl_axes: matplotlib.axes.Axes,
        axes_image: matplotlib.image.AxesImage,
        face_vertices: np.ndarray,
        face_uvs: np.ndarray,
        texture: Texture,
        intensity: np.float64,
        interpolation="none",
    ) -> None:
        """
        Parameters
        ----------
        T : (3,2) np.ndarray
        Positions of the triangle vertices
        UV : (3,2) np.ndarray
        UV coordinates of the triangle vertices
        texture:
        Image to use for texture
        """

        texture_data = texture.data
        image_w, image_h = texture_data.shape[:2]
        uvs_pixel = face_uvs * (image_w, image_h)

        x_min = int(np.floor(uvs_pixel[:, 0].min()))
        x_max = int(np.ceil(uvs_pixel[:, 0].max()))
        y_min = int(np.floor(uvs_pixel[:, 1].min()))
        y_max = int(np.ceil(uvs_pixel[:, 1].max()))

        texture_region = texture_data[y_min:y_max, x_min:x_max, :] * 255.0 * intensity
        texture_region = (texture_region).astype(np.uint8)
        extent = x_min / image_w, x_max / image_w, y_min / image_h, y_max / image_h

        # fake_texture = np.zeros((2, 2, 3), dtype=np.uint8)
        # axes_image = mpl_axes.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))

        matrix_wrap = RendererMeshPhongMaterial.texture_coords_wrap(face_uvs, face_vertices)
        if matrix_wrap is None:
            # if degenerated triangle, hide the image
            axes_image.set_extent((0, 0, 0, 0))
            return

        transform = matrix_wrap + mpl_axes.transData

        path = matplotlib.path.Path(
            [face_uvs[0], face_uvs[1], face_uvs[2], face_uvs[0]],
            closed=True,
        )

        axes_image.set_data(texture_region)
        axes_image.set_interpolation(interpolation)
        axes_image.set_extent(extent)
        axes_image.set_transform(transform)
        axes_image.set_clip_path(path, transform)

    @staticmethod
    def texture_coords_wrap(face_coord_1: np.ndarray, face_coord_2: np.ndarray) -> matplotlib.transforms.Affine2D | None:
        """
        Return an affine transform that warp triangle T1 into triangle
        T2.

        return None if `LinAlgError` if T1 or T2 are degenerated triangles
        """

        face_coord_1 = np.c_[np.array(face_coord_1), np.ones(3)]
        face_coord_2 = np.c_[np.array(face_coord_2), np.ones(3)]
        try:
            matrix = np.linalg.inv(face_coord_1) @ face_coord_2
        except np.linalg.LinAlgError:
            return None

        return matplotlib.transforms.Affine2D(matrix.T)
