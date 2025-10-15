# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.image
import matplotlib.path
import matplotlib.axes
import matplotlib.transforms
import numpy as np


# local imports
from ..objects import Mesh, Scene
from ..core import Texture, Object3D
from .renderer import Renderer
from ..cameras import Camera
from ..lights import Light
from ..materials import MeshTexturedMaterial
from .renderer_utils import RendererUtils


class RendererMeshTexturedMaterial:

    @staticmethod
    def render(
        renderer: "Renderer",
        mesh: Mesh,
        camera: Camera,
        faces_vertices_world: np.ndarray,
        faces_vertices_2d: np.ndarray,
        faces_uvs: np.ndarray,
    ) -> list[matplotlib.artist.Artist]:
        material = typing.cast(MeshTexturedMaterial, mesh.material)

        # =============================================================================
        # Sanity checks
        # =============================================================================

        assert material.texture is not None and material.texture.data.size > 0, "MeshTexturedMaterial requires a valid texture."

        # =============================================================================
        # Face culling
        # =============================================================================

        faces_visible = RendererUtils.compute_faces_visible(faces_vertices_2d, material.face_culling)

        # =============================================================================
        # Lighting - compute faces_color
        # =============================================================================

        # get the scene lights in the scene graph
        scene = mesh.root()
        assert isinstance(scene, Scene)
        lights: list[Light] = [child for child in scene.traverse() if isinstance(child, Light)]

        # compute face normals and centroids in world space
        faces_normals_unit = RendererUtils.compute_faces_normal_unit(faces_vertices_world)
        faces_centroids_world = RendererUtils.compute_faces_centroids(faces_vertices_world)

        # apply flat shading
        faces_color = RendererUtils.shade_faces_flat(
            camera,
            material_color=material.color,
            material_shininess=material.shininess,
            faces_normals_unit=faces_normals_unit,
            faces_centroids_world=faces_centroids_world,
            lights=lights,
        )

        # =============================================================================
        # Compute faces_depth to set zorder
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
        for face_index, (face_vertices_2d, face_uvs, face_color, face_visible, face_depth) in enumerate(
            zip(faces_vertices_2d, faces_uvs, faces_color, faces_visible, faces_depth)
        ):
            # get the artist for this face
            face_uuid = f"{mesh.uuid}_face_{face_index}"
            axes_image = typing.cast(matplotlib.image.AxesImage, renderer._artists[face_uuid])
            changed_artists.append(axes_image)

            # set visibility
            axes_image.set_visible(face_visible)

            # set the zorder based on the depth (the more negative, the closer to the camera), so invert the depth
            # - also, matplotlib has a limited zorder range, so scale it down
            if material.face_sorting:
                axes_image.set_zorder(-face_depth)

            # skip if not visible
            if not face_visible:
                continue

            # update the textured face
            RendererMeshTexturedMaterial.update_textured_face(
                mpl_axes=renderer._axis,
                axes_image=axes_image,
                face_vertices_2d=face_vertices_2d,
                face_uvs=face_uvs,
                texture=material.texture,
                face_color=face_color,
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
        face_color: np.ndarray,
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

        # sanity check
        assert face_vertices_2d.shape == (3, 2), f"face_vertices_2d shape should be (3, 2), got {face_vertices_2d.shape}"
        assert face_uvs.shape == (3, 2), f"face_uvs shape should be (3, 2), got {face_uvs.shape}"
        assert face_color.shape == (3,), f"face_color shape should be (3,), got {face_color.shape}"

        texture_data = texture.data
        image_w, image_h = texture_data.shape[:2]
        uvs_pixel = face_uvs * (image_w, image_h)

        x_min = int(np.floor(uvs_pixel[:, 0].min()))
        x_max = int(np.ceil(uvs_pixel[:, 0].max()))
        y_min = int(np.floor(uvs_pixel[:, 1].min()))
        y_max = int(np.ceil(uvs_pixel[:, 1].max()))

        texture_region = texture_data[y_min:y_max, x_min:x_max, :] * 255.0 * face_color
        texture_region = np.clip(texture_region, 0, 255)
        texture_region = (texture_region).astype(np.uint8)
        extent = x_min / image_w, x_max / image_w, y_min / image_h, y_max / image_h

        # fake_texture = np.zeros((2, 2, 3), dtype=np.uint8)
        # axes_image = mpl_axes.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))

        matrix_wrap = RendererMeshTexturedMaterial.texture_coords_wrap(face_uvs, face_vertices_2d)
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
