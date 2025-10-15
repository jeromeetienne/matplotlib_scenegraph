# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import numpy as np
from pyrr import vector, vector3

# local imports
from ..objects import Mesh, Scene
from ..lights import Light, DirectionalLight, PointLight, AmbientLight
from .renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core import Object3D
from ..materials import MeshPhongMaterial
from .renderer_mesh import RendererMesh
from .renderer_utils import RendererUtils


class RendererMeshPhongMaterial:

    # =============================================================================
    #
    # =============================================================================

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
        # Compute faces_color
        # =============================================================================

        # get the scene lights in the scene graph
        scene = mesh.root()
        assert isinstance(scene, Scene)
        lights: list[Light] = [child for child in scene.traverse() if isinstance(child, Light)]

        # compute face normals and centroids in world space
        faces_normals_unit = RendererUtils.compute_faces_normal_unit(faces_vertices_world)
        faces_centroids_world = RendererUtils.compute_faces_centroids(faces_vertices_world)

        # apply flat shading
        shaded_colors = RendererUtils.shade_faces_flat(camera, material.color, material.shininess, faces_normals_unit, faces_centroids_world, lights)

        # apply vertex colors if any
        faces_color = shaded_colors

        # =============================================================================
        # Honor material.face_sorting
        # =============================================================================

        # Sort polygons by depth (painter's algorithm)
        # - faces are sorted based on their depth (z) in camera space within a single artist
        # - this artist.set_zorder() is set based on the distance from the camera to the Object3D position
        # - so possible conflict between faces of different objects
        # - CAUTION: here reorder ALL arrays you use below to keep them in sync
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
        # honor material.face_culling
        # =============================================================================

        faces_visible = RendererUtils.compute_faces_visible(faces_vertices_2d, material.face_culling)

        # keep only visible faces
        # - CAUTION: here reorder ALL arrays you use below to keep them in sync
        faces_vertices_2d = faces_vertices_2d[faces_visible]
        faces_color = faces_color[faces_visible]

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
        # do z-ordering based on distance to camera
        # =============================================================================

        # compute and set zorder on our single artist
        RendererUtils.update_single_artist_zorder(camera, mesh, mpl_poly_collection)

        # =============================================================================
        # Update all the artists
        # =============================================================================

        # update the PathCollection with the new patches
        mpl_poly_collection.set_verts(typing.cast(list, faces_vertices_2d))
        mpl_poly_collection.set_facecolor(typing.cast(list, faces_color))
        mpl_poly_collection.set_edgecolor((typing.cast(list, material.edge_colors)))
        mpl_poly_collection.set_linewidth((typing.cast(list, material.edge_widths)))

        return [mpl_poly_collection]
