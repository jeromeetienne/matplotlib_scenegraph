# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import numpy as np


# local imports
from ..materials import MeshBasicMaterial
from ..objects.mesh import Mesh
from .renderer import Renderer
from ..cameras.camera import Camera
from .renderer_utils import RendererUtils


class RendererMeshBasicMaterial:

    @staticmethod
    def render(
        renderer: "Renderer",
        mesh: Mesh,
        camera: Camera,
        faces_vertices_ndc: np.ndarray,
        faces_vertices_2d: np.ndarray,
    ) -> list[matplotlib.artist.Artist]:
        geometry = mesh.geometry
        material = typing.cast(MeshBasicMaterial, mesh.material)

        # =============================================================================
        # Sanity checks
        # =============================================================================

        # faces_vertices_ndc: (n_faces, 3, 3) array of the 3D vertices of each face in NDC space
        # faces_vertices_2d: (n_faces, 3, 2) array of the 2D vertices of each face in screen space
        # faces_uvs: (n_faces, 3, 2) array of the UV coordinates of each face

        assert isinstance(material, MeshBasicMaterial), f"Expected material to be a MeshBasicMaterial, got {type(material)}"
        assert faces_vertices_ndc.shape == (
            len(geometry.indices),
            3,
            3,
        ), f"Expected faces_vertices_ndc to have shape {(len(geometry.indices), 3, 3)}, got {faces_vertices_ndc.shape}"
        assert faces_vertices_2d.shape == (
            len(geometry.indices),
            3,
            2,
        ), f"Expected faces_vertices_2d to have shape {(len(geometry.indices), 3, 2)}, got {faces_vertices_2d.shape}"
        assert len(faces_vertices_2d) == len(
            geometry.indices
        ), f"Expected faces_vertices_2d to have {len(geometry.indices)} faces, got {len(faces_vertices_2d)}"

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

        # =============================================================================
        # honor material.face_culling
        # =============================================================================

        faces_visible = RendererUtils.compute_faces_visible(faces_vertices_2d, material.face_culling)
        # print(f"faces_visible: {faces_visible.sum()}/{len(faces_visible)}")

        # remove hidden faces
        faces_vertices_2d = faces_vertices_2d[faces_visible]

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
        mpl_poly_collection.set_facecolor(typing.cast(list, material.colors))
        mpl_poly_collection.set_edgecolor(typing.cast(list, material.edge_colors))
        mpl_poly_collection.set_linewidth(typing.cast(list, material.edge_widths))

        return [mpl_poly_collection]
