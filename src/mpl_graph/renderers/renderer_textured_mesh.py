# stdlib imports
import typing
from typing import Sequence

# pip imports
import matplotlib.artist
import matplotlib.image
import matplotlib.path
import matplotlib.axes
import matplotlib.transforms
import numpy as np

# local imports
from ..objects.textured_mesh import TexturedMesh
from ..renderers.renderer import RendererMatplotlib
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils


class MatplotlibRendererTexturedMesh:
    @staticmethod
    def render(renderer: "RendererMatplotlib", textured_mesh: TexturedMesh, camera: CameraBase) -> Sequence[matplotlib.artist.Artist]:
        mpl_axes: matplotlib.axes.Axes = renderer.get_axis()
        mesh_position = textured_mesh.position.copy()
        camera_position = camera.position.copy()
        light_position = camera_position  # for now the light is at the camera position

        assert mesh_position.shape == (3,)
        assert camera_position.shape == (3,)
        assert light_position.shape == (3,)

        mesh = textured_mesh

        faces_vertices = mesh.faces_vertices.copy()
        faces_uvs = mesh.faces_uvs.copy()
        texture = mesh.texture_data

        # =============================================================================
        #
        # =============================================================================
        # Create a list of axes images for each face
        axes_images: list[matplotlib.image.AxesImage] = []
        faces_count = len(textured_mesh.faces_vertices)
        for _ in range(faces_count):
            fake_texture = np.zeros((2, 2, 3), dtype=np.uint8)
            axes_image = mpl_axes.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))
            axes_images.append(axes_image)

        # =============================================================================
        # Compute face normals - needed for lighting and back-face culling
        # =============================================================================
        faces_normals = np.cross(
            faces_vertices[:, 2] - faces_vertices[:, 0],
            faces_vertices[:, 1] - faces_vertices[:, 0],
        )
        faces_normals_unit = faces_normals / np.linalg.norm(faces_normals, axis=1).reshape(len(faces_normals), 1)

        # =============================================================================
        # Face culling
        # =============================================================================

        # camera_cosines is the cosine of the angle between the normal and the camera
        # camera_direction = (0, 0, -1)
        camera_direction = camera_position - mesh_position
        camera_cosines: np.ndarray = np.dot(faces_normals_unit, camera_direction)

        faces_hidden = camera_cosines <= 0

        # back face culling
        # faces_vertices = faces_vertices[camera_cosines > 0]
        # faces_uvs = faces_uvs[camera_cosines > 0]
        # faces_normals_unit = faces_normals_unit[camera_cosines > 0]

        # =============================================================================
        # Lighting
        # =============================================================================
        light_direction = light_position - mesh_position
        light_direction_unit = light_direction / np.linalg.norm(light_direction)
        light_cosines: np.ndarray = np.dot(faces_normals_unit, light_direction_unit)
        light_intensities = (light_cosines + 1) / 2

        # =============================================================================
        # Sort triangles by depth (painter's algorithm)
        # =============================================================================
        faces_depth = faces_vertices[:, :, 2].mean(axis=1)
        depth_sorted_indices = np.argsort(faces_depth)
        faces_vertices = faces_vertices[depth_sorted_indices][..., :2]
        faces_uvs = faces_uvs[depth_sorted_indices][..., :2]
        light_intensities = light_intensities[depth_sorted_indices]
        faces_hidden = faces_hidden[depth_sorted_indices]

        # =============================================================================
        # Loop over faces and draw them
        # =============================================================================
        for face_index, (face_vertices, face_uvs, light_intensity, face_hidden) in enumerate(zip(faces_vertices, faces_uvs, light_intensities, faces_hidden)):
            if face_hidden:
                continue
            axes_image = axes_images[face_index]
            MatplotlibRendererTexturedMesh.update_textured_triangle(
                mpl_axes=mpl_axes,
                axes_image=axes_image,
                face_vertices=face_vertices,
                face_uvs=face_uvs,
                texture=texture,
                intensity=light_intensity,
            )

        return axes_images

    @staticmethod
    def update_textured_triangle(
        mpl_axes: matplotlib.axes.Axes,
        axes_image: matplotlib.image.AxesImage,
        face_vertices: np.ndarray,
        face_uvs: np.ndarray,
        texture: np.ndarray,
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

        image_w, image_h = texture.shape[:2]
        uvs_pixel = face_uvs * (image_w, image_h)

        x_min = int(np.floor(uvs_pixel[:, 0].min()))
        x_max = int(np.ceil(uvs_pixel[:, 0].max()))
        y_min = int(np.floor(uvs_pixel[:, 1].min()))
        y_max = int(np.ceil(uvs_pixel[:, 1].max()))

        texture = (texture[y_min:y_max, x_min:x_max, :] * intensity).astype(np.uint8)
        extent = x_min / image_w, x_max / image_w, y_min / image_h, y_max / image_h

        # fake_texture = np.zeros((2, 2, 3), dtype=np.uint8)
        # axes_image = mpl_axes.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))

        matrix_wrap = MatplotlibRendererTexturedMesh.texture_coords_wrap(face_uvs, face_vertices)
        if matrix_wrap is None:
            # if degenerated triangle, hide the image
            axes_image.set_extent((0, 0, 0, 0))
            return

        transform = matrix_wrap + mpl_axes.transData

        path = matplotlib.path.Path(
            [face_uvs[0], face_uvs[1], face_uvs[2], face_uvs[0]],
            closed=True,
        )

        axes_image.set_data(texture)
        axes_image.set_interpolation(interpolation)
        axes_image.set_extent(extent)
        axes_image.set_transform(transform)
        axes_image.set_clip_path(path, transform)

    # =============================================================================
    # Affine transform to warp triangles
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
