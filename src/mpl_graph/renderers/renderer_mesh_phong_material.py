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

        if material.texture is None:
            from .renderer_mesh_phong_material_notexture import RendererMeshPhongMaterialNoTexture

            changed_artists = RendererMeshPhongMaterialNoTexture.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_world=faces_vertices_world,
                faces_vertices_2d=faces_vertices_2d,
                light_intensities=light_intensities,
                faces_visible=faces_visible,
            )
            return changed_artists
        else:
            from .renderer_mesh_phong_material_texture import RendererMeshPhongMaterialTexture

            changed_artists = RendererMeshPhongMaterialTexture.render(
                renderer=renderer,
                mesh=mesh,
                camera=camera,
                faces_vertices_world=faces_vertices_world,
                faces_vertices_2d=faces_vertices_2d,
                faces_uvs=faces_uvs,
                texture=material.texture,
                light_intensities=light_intensities,
                faces_visible=faces_visible,
                # faces_depth=faces_depth,
            )
            return changed_artists
