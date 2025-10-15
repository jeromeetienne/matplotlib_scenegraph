# stdlib imports
import typing

# pip imports
import matplotlib.artist
import numpy as np


# local imports
from ..core.constants import Constants
from ..objects.mesh import Mesh
from ..materials import MeshBasicMaterial, MeshPhongMaterial, MeshNormalMaterial, MeshDepthMaterial, MeshTexturedMaterial
from .renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..math.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils

# https://chatgpt.com/c/68ee0eab-776c-8331-b44a-f131ba3f166b
# local -> world -> view -> clip (NDC) -> screen (2D)

# local -> world : coord in world space
# world -> view : coord in camera space
# view -> clip (NDC) : coord in normalized device coord space
# clip (NDC) -> screen (2D) : coord in 2D screen


class RendererMesh:
    @staticmethod
    def render(renderer: "Renderer", mesh: Mesh, camera: CameraBase) -> list[matplotlib.artist.Artist]:

        # =============================================================================
        # sanity checks
        # =============================================================================

        assert mesh.geometry.indices is not None, "The mesh geometry must have face indices to be rendered"
        assert mesh.geometry.uvs is not None, "The mesh geometry must have texture coordinates to be rendered"

        # =============================================================================
        # Extract geometry and material
        # =============================================================================

        geometry = mesh.geometry
        material = mesh.material
        faces_uvs = mesh.geometry.uvs[geometry.indices]

        # =============================================================================
        # Compute the world space faces_vertices
        # =============================================================================

        # Get the full transform matrix for the mesh
        world_matrix = mesh.get_world_matrix()
        vertices_world = GeometryUtils.apply_transform(geometry.vertices, world_matrix)

        # build the faces vertices and uvs arrays
        faces_vertices_world = vertices_world[geometry.indices]

        # =============================================================================
        # Compute the NDC faces_vertices
        # =============================================================================

        # Get the full transform matrix for the mesh
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, mesh)
        vertices_ndc, vertices_clip = GeometryUtils.apply_mvp_matrix(geometry.vertices, mvp_matrix)

        # build the faces vertices and uvs arrays
        faces_vertices_ndc = vertices_ndc[geometry.indices]

        # =============================================================================
        # Switch vertices to 2d
        # =============================================================================

        # drop z for 2D rendering
        faces_vertices_2d = faces_vertices_ndc[..., :2]

        # =============================================================================
        # Render the mesh using the appropriate material
        # =============================================================================

        if isinstance(mesh.material, MeshBasicMaterial):
            from .renderer_mesh_basic_material import RendererMeshBasicMaterial

            changed_artists = RendererMeshBasicMaterial.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_ndc=faces_vertices_ndc,
                faces_vertices_2d=faces_vertices_2d,
            )
        elif isinstance(mesh.material, MeshNormalMaterial):
            from .renderer_mesh_normal_material import RendererMeshNormalMaterial

            changed_artists = RendererMeshNormalMaterial.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_world=faces_vertices_world,
                faces_vertices_ndc=faces_vertices_ndc,
                faces_vertices_2d=faces_vertices_2d,
            )
        elif isinstance(mesh.material, MeshDepthMaterial):
            from .renderer_mesh_depth_material import RendererMeshDepthMaterial

            changed_artists = RendererMeshDepthMaterial.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_ndc=faces_vertices_ndc,
                faces_vertices_2d=faces_vertices_2d,
            )
        elif isinstance(mesh.material, MeshPhongMaterial):
            from .renderer_mesh_phong_material import RendererMeshPhongMaterial

            changed_artists = RendererMeshPhongMaterial.render(
                renderer=renderer,
                camera=camera,
                mesh=mesh,
                faces_vertices_world=faces_vertices_world,
                faces_vertices_ndc=faces_vertices_ndc,
                faces_vertices_2d=faces_vertices_2d,
            )
        elif isinstance(mesh.material, MeshTexturedMaterial):
            from .renderer_mesh_textured_material import RendererMeshTexturedMaterial

            changed_artists = RendererMeshTexturedMaterial.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_world=faces_vertices_world,
                faces_vertices_2d=faces_vertices_2d,
                faces_uvs=faces_uvs,
            )
        else:
            raise ValueError(f"Unsupported material type: {type(mesh.material)}")
        return changed_artists
