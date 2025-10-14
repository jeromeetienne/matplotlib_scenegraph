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
from .renderer_mesh_phong_material import RendererMeshPhongMaterial
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils


class RendererMesh:
    @staticmethod
    def render(renderer: "Renderer", mesh: Mesh, camera: CameraBase) -> list[matplotlib.artist.Artist]:

        assert mesh.geometry.indices is not None, "The mesh geometry must have face indices to be rendered"
        assert mesh.geometry.uvs is not None, "The mesh geometry must have texture coordinates to be rendered"

        geometry = mesh.geometry
        material = mesh.material
        faces_uvs = mesh.geometry.uvs[geometry.indices]

        # =============================================================================
        #
        # =============================================================================

        # Get the full transform matrix for the mesh
        world_matrix = mesh.get_world_matrix()
        vertices_world = GeometryUtils.apply_transform(geometry.vertices, world_matrix)

        # build the faces vertices and uvs arrays
        faces_vertices_world = vertices_world[geometry.indices]

        # =============================================================================
        # Compute face normals - needed for lighting and back-face culling
        # =============================================================================
        faces_normals = np.cross(
            faces_vertices_world[:, 2] - faces_vertices_world[:, 0],
            faces_vertices_world[:, 1] - faces_vertices_world[:, 0],
        )
        faces_normals_unit = faces_normals / np.linalg.norm(faces_normals, axis=1).reshape(len(faces_normals), 1)
        """Compute the face normals for lighting and culling. shape = [num_faces, 3] in world space"""

        # =============================================================================
        # Face culling
        # =============================================================================

        # camera_cosines is the cosine of the angle between the normal and the camera
        # - if <= 0, the face is pointing away from the camera
        # - if > 0, the face is pointing towards the camera
        # camera_position = camera.get_world_position()
        # mesh_position = mesh.get_world_position()
        # camera_direction = (0, 0, -1)
        camera_direction = mesh.get_world_position() - camera.get_world_position()
        # camera_direction /= np.linalg.norm(camera_direction)
        camera_cosines: np.ndarray = np.dot(faces_normals_unit, camera_direction)

        # faces_hidden = camera_cosines <= 0
        # faces_hidden = np.zeros(len(faces_vertices_world), dtype=bool)

        # NOTE: dont remove the faces, just hide them later
        # - required for keeping the artist list stable

        # =============================================================================
        # Lighting
        # =============================================================================
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
        # Compute the NDC faces_vertices
        # =============================================================================

        # Get the full transform matrix for the mesh
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, mesh)
        vertices_ndc = GeometryUtils.apply_transform(geometry.vertices, mvp_matrix)

        # build the faces vertices and uvs arrays
        faces_vertices_ndc = vertices_ndc[geometry.indices]

        # drop z for 2D rendering
        faces_vertices_2d = faces_vertices_ndc[..., :2]

        # =============================================================================
        # Face Culling
        # =============================================================================

        # For each face, compute the cross product of the edges in 2D
        # - if the z component is positive, the face is oriented counter-clockwise
        # - if the z component is negative, the face is oriented clockwise
        # - if the z component is zero, the face is degenerated (line or point)
        faces_edges_2d_a = faces_vertices_2d[:, 1] - faces_vertices_2d[:, 0]
        faces_edges_2d_b = faces_vertices_2d[:, 2] - faces_vertices_2d[:, 0]
        faces_cross_z = faces_edges_2d_a[:, 0] * faces_edges_2d_b[:, 1] - faces_edges_2d_a[:, 1] * faces_edges_2d_b[:, 0]
        # this is the threshold below which a face is considered degenerated and trigger exception when inverting matrix
        faces_cross_threshold = 1e-6
        if material.face_culling == Constants.FaceCulling.FrontSide:
            faces_visible = faces_cross_z <= -faces_cross_threshold
        elif material.face_culling == Constants.FaceCulling.BackSide:
            faces_visible = faces_cross_z >= faces_cross_threshold
        elif material.face_culling == Constants.FaceCulling.BothSides:
            # If the face is degenerated (line or point), it is not visible
            faces_visible = np.abs(faces_cross_z) > faces_cross_threshold
        else:
            raise ValueError(f"Unknown face culling mode: {material.face_culling}")

        print(f"faces_visible: {faces_visible.sum()}/{len(faces_visible)}")

        # =============================================================================
        # Create the artists if needed
        # =============================================================================
        face_uuid = f"{mesh.uuid}_face_0"
        if face_uuid not in renderer._artists:
            # Create a list of axes images for each face
            faces_count = len(faces_vertices_world)
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
                face_vertices=face_vertices_2d,
                face_uvs=face_uvs,
                texture=material.texture,
                intensity=light_intensity,
            )

        return changed_artists

    # # =============================================================================
    # # Update the artist for a texture face
    # # =============================================================================
    # @staticmethod
    # def update_textured_face(
    #     mpl_axes: matplotlib.axes.Axes,
    #     axes_image: matplotlib.image.AxesImage,
    #     face_vertices: np.ndarray,
    #     face_uvs: np.ndarray,
    #     texture: Texture,
    #     intensity: np.float64,
    #     interpolation="none",
    # ) -> None:
    #     """
    #     Parameters
    #     ----------
    #     T : (3,2) np.ndarray
    #     Positions of the triangle vertices
    #     UV : (3,2) np.ndarray
    #     UV coordinates of the triangle vertices
    #     texture:
    #     Image to use for texture
    #     """

    #     texture_data = texture.data
    #     image_w, image_h = texture_data.shape[:2]
    #     uvs_pixel = face_uvs * (image_w, image_h)

    #     x_min = int(np.floor(uvs_pixel[:, 0].min()))
    #     x_max = int(np.ceil(uvs_pixel[:, 0].max()))
    #     y_min = int(np.floor(uvs_pixel[:, 1].min()))
    #     y_max = int(np.ceil(uvs_pixel[:, 1].max()))

    #     texture_region = texture_data[y_min:y_max, x_min:x_max, :] * 255.0 * intensity
    #     texture_region = (texture_region).astype(np.uint8)
    #     extent = x_min / image_w, x_max / image_w, y_min / image_h, y_max / image_h

    #     # fake_texture = np.zeros((2, 2, 3), dtype=np.uint8)
    #     # axes_image = mpl_axes.imshow(fake_texture, origin="lower", extent=(0, 0, 0, 0))

    #     matrix_wrap = RendererMesh.texture_coords_wrap(face_uvs, face_vertices)
    #     if matrix_wrap is None:
    #         # if degenerated triangle, hide the image
    #         axes_image.set_extent((0, 0, 0, 0))
    #         return

    #     transform = matrix_wrap + mpl_axes.transData

    #     path = matplotlib.path.Path(
    #         [face_uvs[0], face_uvs[1], face_uvs[2], face_uvs[0]],
    #         closed=True,
    #     )

    #     axes_image.set_data(texture_region)
    #     axes_image.set_interpolation(interpolation)
    #     axes_image.set_extent(extent)
    #     axes_image.set_transform(transform)
    #     axes_image.set_clip_path(path, transform)

    # @staticmethod
    # def texture_coords_wrap(face_coord_1: np.ndarray, face_coord_2: np.ndarray) -> matplotlib.transforms.Affine2D | None:
    #     """
    #     Return an affine transform that warp triangle T1 into triangle
    #     T2.

    #     return None if `LinAlgError` if T1 or T2 are degenerated triangles
    #     """

    #     face_coord_1 = np.c_[np.array(face_coord_1), np.ones(3)]
    #     face_coord_2 = np.c_[np.array(face_coord_2), np.ones(3)]
    #     try:
    #         matrix = np.linalg.inv(face_coord_1) @ face_coord_2
    #     except np.linalg.LinAlgError:
    #         return None

    #     return matplotlib.transforms.Affine2D(matrix.T)
