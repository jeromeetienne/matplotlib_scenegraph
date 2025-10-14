# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import numpy as np

# local imports
from ..objects.mesh import Mesh
from .renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core import Object3D
from ..materials import MeshPhongMaterial
from .renderer_mesh import RendererMesh


class RendererMeshPhongMaterial:

    @staticmethod
    def render(
        renderer: "Renderer",
        mesh: Mesh,
        camera: CameraBase,
        faces_vertices_world: np.ndarray,
        faces_vertices_ndc: np.ndarray,
        faces_vertices_2d: np.ndarray,
    ) -> list[matplotlib.artist.Artist]:
        material = typing.cast(MeshPhongMaterial, mesh.material)

        # =============================================================================
        # Lighting
        # =============================================================================

        faces_normals_unit = RendererMesh.compute_faces_normal_unit(faces_vertices_world)

        # light_direction = light_position - mesh_position
        light_direction = np.array((1.0, 1.0, 1.0)).astype(np.float32)
        light_direction /= np.linalg.norm(light_direction)
        light_cosines: np.ndarray = np.dot(faces_normals_unit, light_direction)
        light_intensities = (light_cosines + 1) / 2

        faces_color = np.zeros((len(faces_vertices_2d), 4))
        faces_color[:, 0] = material.color[0] * light_intensities
        faces_color[:, 1] = material.color[1] * light_intensities
        faces_color[:, 2] = material.color[2] * light_intensities
        faces_color[:, 3] = 1.0  # alpha

        # =============================================================================
        # Depth sort at the faces level
        # =============================================================================

        # Sort polygons by depth (painter's algorithm)
        # FIXME how does it interact with the zorder at the object3D level ?
        if material.face_sorting:
            # compute the depth of each face as the mean z value of its vertices
            faces_depth = faces_vertices_ndc[:, :, 2].mean(axis=1)
            # get the sorting indices (from farthest to nearest)
            depth_sorted_indices = np.argsort(faces_depth)
            # apply the sorting to faces_vertices and faces_hidden
            # CAUTION: here reorder ALL arrays you use below to keep them in sync
            faces_vertices_2d = faces_vertices_2d[depth_sorted_indices]
            faces_color = faces_color[depth_sorted_indices]

        # =============================================================================
        # Face culling
        # =============================================================================

        faces_visible = RendererMesh.compute_faces_visible(faces_vertices_2d, material.face_culling)

        # keep only visible faces
        # - CAUTION: here reorder ALL arrays you use below to keep them in sync
        faces_vertices_2d = faces_vertices_2d[faces_visible]
        faces_color = faces_color[faces_visible]

        # =============================================================================
        #
        # =============================================================================

        # =============================================================================
        #
        # =============================================================================

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
        # TODO handle that and put them in the material
        mpl_poly_collection.set_edgecolor((0, 0, 0, 0.3))
        mpl_poly_collection.set_linewidth(0.5)

        return [mpl_poly_collection]
