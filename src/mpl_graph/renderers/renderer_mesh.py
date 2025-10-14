# stdlib imports
import typing

# pip imports
import matplotlib.artist
import numpy as np


# local imports
from ..core.constants import Constants
from ..objects.mesh import Mesh
from ..materials import MeshBasicMaterial, MeshPhongMaterial
from .renderer import Renderer
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
        # camera_direction = mesh.get_world_position() - camera.get_world_position()
        # # camera_direction /= np.linalg.norm(camera_direction)
        # camera_cosines: np.ndarray = np.dot(faces_normals_unit, camera_direction)

        # faces_hidden = camera_cosines <= 0
        # faces_hidden = np.zeros(len(faces_vertices_world), dtype=bool)

        # NOTE: dont remove the faces, just hide them later
        # - required for keeping the artist list stable

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

        # # For each face, compute the cross product of the edges in 2D
        # # - if the z component is positive, the face is oriented counter-clockwise
        # # - if the z component is negative, the face is oriented clockwise
        # # - if the z component is zero, the face is degenerated (line or point)
        # faces_edges_2d_a = faces_vertices_2d[:, 1] - faces_vertices_2d[:, 0]
        # faces_edges_2d_b = faces_vertices_2d[:, 2] - faces_vertices_2d[:, 0]
        # faces_cross_z = faces_edges_2d_a[:, 0] * faces_edges_2d_b[:, 1] - faces_edges_2d_a[:, 1] * faces_edges_2d_b[:, 0]
        # # this is the threshold below which a face is considered degenerated and trigger exception when inverting matrix
        # faces_cross_threshold = 1e-6
        # if material.face_culling == Constants.FaceCulling.FrontSide:
        #     faces_visible = faces_cross_z <= -faces_cross_threshold
        # elif material.face_culling == Constants.FaceCulling.BackSide:
        #     faces_visible = faces_cross_z >= faces_cross_threshold
        # elif material.face_culling == Constants.FaceCulling.BothSides:
        #     # If the face is degenerated (line or point), it is not visible
        #     faces_visible = np.abs(faces_cross_z) > faces_cross_threshold
        # else:
        #     raise ValueError(f"Unknown face culling mode: {material.face_culling}")

        # =============================================================================
        #
        # =============================================================================

        if isinstance(material, MeshBasicMaterial):
            from .renderer_mesh_basic_material import RendererMeshBasicMaterial

            changed_artists = RendererMeshBasicMaterial.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_ndc=faces_vertices_ndc,
                faces_vertices_2d=faces_vertices_2d,
                faces_uvs=faces_uvs,
            )
        elif isinstance(material, MeshPhongMaterial):
            from .renderer_mesh_phong_material import RendererMeshPhongMaterial

            faces_visible = RendererMesh.compute_faces_visible(faces_vertices_2d, material.face_culling)
            print(f"faces_visible: {faces_visible.sum()}/{len(faces_visible)}")
            # =============================================================================
            # Lighting
            # =============================================================================
            # light_direction = light_position - mesh_position
            light_direction = np.array((1.0, 1.0, 1.0)).astype(np.float32)
            light_direction /= np.linalg.norm(light_direction)
            light_cosines: np.ndarray = np.dot(faces_normals_unit, light_direction)
            light_intensities = (light_cosines + 1) / 2

            changed_artists = RendererMeshPhongMaterial.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_world=faces_vertices_world,
                faces_vertices_2d=faces_vertices_2d,
                faces_uvs=faces_uvs,
                light_intensities=light_intensities,
                faces_visible=faces_visible,
            )
        else:
            raise ValueError(f"Unsupported material type: {type(material)}")
        return changed_artists

    # =============================================================================
    #
    # =============================================================================
    @staticmethod
    def compute_faces_visible(faces_vertices_2d: np.ndarray, face_culling: Constants.FaceCulling) -> np.ndarray:
        """Compute which faces are visible based on their normals and the camera position.

        Returns:
            np.ndarray: A boolean array indicating which faces are visible.
        """

        # For each face, compute the cross product of the edges in 2D
        # - if the z component is positive, the face is oriented counter-clockwise
        # - if the z component is negative, the face is oriented clockwise
        # - if the z component is zero, the face is degenerated (line or point)
        faces_edges_2d_a = faces_vertices_2d[:, 1] - faces_vertices_2d[:, 0]
        faces_edges_2d_b = faces_vertices_2d[:, 2] - faces_vertices_2d[:, 0]
        faces_cross_z = faces_edges_2d_a[:, 0] * faces_edges_2d_b[:, 1] - faces_edges_2d_a[:, 1] * faces_edges_2d_b[:, 0]
        # this is the threshold below which a face is considered degenerated and trigger exception when inverting matrix
        faces_cross_threshold = 1e-6
        if face_culling == Constants.FaceCulling.FrontSide:
            faces_visible = faces_cross_z <= -faces_cross_threshold
        elif face_culling == Constants.FaceCulling.BackSide:
            faces_visible = faces_cross_z >= faces_cross_threshold
        elif face_culling == Constants.FaceCulling.BothSides:
            # If the face is degenerated (line or point), it is not visible
            faces_visible = np.abs(faces_cross_z) > faces_cross_threshold
        else:
            raise ValueError(f"Unknown face culling mode: {face_culling}")

        print(f"faces_visible: {faces_visible.sum()}/{len(faces_visible)}")
        return faces_visible
