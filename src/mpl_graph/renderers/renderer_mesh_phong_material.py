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
from ..materials import MeshPhongMaterial
from .renderer_mesh import RendererMesh


class RendererMeshPhongMaterial:

    @staticmethod
    def render(
        renderer: "Renderer",
        mesh: Mesh,
        camera: CameraBase,
        faces_vertices_world: np.ndarray,
        faces_vertices_2d: np.ndarray,
        faces_uvs: np.ndarray,
    ) -> list[matplotlib.artist.Artist]:
        material = typing.cast(MeshPhongMaterial, mesh.material)

        # =============================================================================
        # Face culling
        # =============================================================================
        faces_visible = RendererMesh.compute_faces_visible(faces_vertices_2d, material.face_culling)
        print(f"faces_visible: {faces_visible.sum()}/{len(faces_visible)}")

        # =============================================================================
        # Lighting
        # =============================================================================

        faces_normals_unit = RendererMesh.compute_faces_normal_unit(faces_vertices_world)

        # light_direction = light_position - mesh_position
        light_direction = np.array((1.0, 1.0, 1.0)).astype(np.float32)
        light_direction /= np.linalg.norm(light_direction)
        light_cosines: np.ndarray = np.dot(faces_normals_unit, light_direction)
        light_intensities = (light_cosines + 1) / 2

        # =============================================================================
        # Sort triangles by depth (painter's algorithm)
        # =============================================================================

        # compute face depth as the mean Z of the face vertices in world space (negative is in front of the camera)
        # - will be used for zorder in matplotlib
        faces_depth = faces_vertices_world[:, :, 2].mean(axis=1)

        # =============================================================================
        # Create the artists if needed
        # =============================================================================
        face_uuid = f"{mesh.uuid}_face_0"
        if face_uuid not in renderer._artists:
            # Create a list of axes images for each face
            faces_count = len(faces_vertices_2d)
            fake_texture = np.zeros((1, 1, 3), dtype=np.uint8)
            for face_index in range(faces_count):
                axes_image = renderer._axis.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))
                axes_image.set_visible(False)  # hide until properly positioned and sized
                face_uuid = f"{mesh.uuid}_face_{face_index}"
                renderer._artists[face_uuid] = axes_image

        # =============================================================================
        # Loop over faces and draw them
        # =============================================================================
        changed_artists: list[matplotlib.artist.Artist] = []
        for face_index, (face_vertices_2d, face_uvs, light_intensity, face_visible, face_depth) in enumerate(
            zip(faces_vertices_2d, faces_uvs, light_intensities, faces_visible, faces_depth)
        ):
            face_uuid = f"{mesh.uuid}_face_{face_index}"
            changed_artist = renderer._artists[face_uuid]
            changed_artists.append(changed_artist)

            # set the zorder based on the depth (the more negative, the closer to the camera), so invert the depth
            # - also, matplotlib has a limited zorder range, so scale it down
            # if material.face_sorting:
            changed_artist.set_zorder(-face_depth)
            changed_artist.set_visible(face_visible)

            # skip hidden faces
            if not face_visible:
                continue

            axes_image = typing.cast(matplotlib.image.AxesImage, changed_artist)
            RendererMeshPhongMaterial.update_textured_face(
                mpl_axes=renderer._axis,
                axes_image=axes_image,
                face_vertices_2d=face_vertices_2d,
                face_uvs=face_uvs,
                texture=material.texture,
                intensity=light_intensity,
            )

        return changed_artists

    # =============================================================================
    # Update the artist for a texture face
    # =============================================================================
    @staticmethod
    def update_textured_face(
        mpl_axes: matplotlib.axes.Axes,
        axes_image: matplotlib.image.AxesImage,
        face_vertices_2d: np.ndarray,
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

        matrix_wrap = RendererMeshPhongMaterial.texture_coords_wrap(face_uvs, face_vertices_2d)
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

    # =============================================================================
    # Compute mapping between image coord and screen coordinate
    # =============================================================================

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
