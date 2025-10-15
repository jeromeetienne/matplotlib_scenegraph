# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import matplotlib.pyplot
import numpy as np

# local imports
from ..materials import MeshDepthMaterial
from ..objects.mesh import Mesh
from .renderer import Renderer
from ..cameras.camera_base import CameraBase
from .renderer_mesh import RendererMesh
from .renderer_utils import RendererUtils


class RendererMeshDepthMaterial:

    @staticmethod
    def render(
        renderer: "Renderer",
        mesh: Mesh,
        camera: CameraBase,
        faces_vertices_world: np.ndarray,
        faces_vertices_ndc: np.ndarray,
        faces_vertices_2d: np.ndarray,
    ) -> list[matplotlib.artist.Artist]:
        geometry = mesh.geometry
        material = typing.cast(MeshDepthMaterial, mesh.material)

        # faces_vertices_world: (n_faces, 3, 3) array of the 3D vertices of each face in world space
        # faces_vertices_2d: (n_faces, 3, 2) array of the 2D vertices of each face in screen space
        # faces_uvs: (n_faces, 3, 2) array of the UV coordinates of each face

        # sanity check
        assert isinstance(material, MeshDepthMaterial), f"Expected material to be a MeshDepthMaterial, got {type(material)}"
        assert faces_vertices_ndc.shape == (
            len(geometry.indices),
            3,
            3,
        ), f"Expected faces_vertices_world to have shape {(len(geometry.indices), 3, 3)}, got {faces_vertices_ndc.shape}"
        assert faces_vertices_2d.shape == (
            len(geometry.indices),
            3,
            2,
        ), f"Expected faces_vertices_2d to have shape {(len(geometry.indices), 3, 2)}, got {faces_vertices_2d.shape}"
        assert len(faces_vertices_2d) == len(
            geometry.indices
        ), f"Expected faces_vertices_2d to have {len(geometry.indices)} faces, got {len(faces_vertices_2d)}"

        # =============================================================================
        # Computes face_colors
        # =============================================================================

        # # faces_depth has a shape of (n_faces,) with the mean Z of each face in NDC space
        faces_depth_ndc = faces_vertices_ndc[:, :, 2].mean(axis=1)

        # normalize faces_depth to be between 0 and 1
        # face_depth_normalized has a shape of (n_faces,) with values between 0 and 1
        depth_min = faces_depth_ndc.min()
        depth_max = faces_depth_ndc.max()
        brightness = (faces_depth_ndc - depth_min) / (depth_max - depth_min)

        # in theory _ndc is between -1 and +1 on all axis, so i could do (ndc + 1 / 2) to normalize it
        # but my ndc is wrong :(
        # brightness = (faces_depth_ndc + 1.0) / 2.0

        # map brightness to a colormap to get RGBA colors
        color_map = matplotlib.pyplot.get_cmap(material.colormap_name)
        colors_rgba = color_map(brightness)
        faces_color = colors_rgba

        # =============================================================================
        # honor material.face_sorting
        # =============================================================================

        # compute face depth as the mean Z of the face vertices in world space (negative is in front of the camera)
        # - will be used for zorder in matplotlib
        if material.face_sorting:
            faces_depth = faces_vertices_ndc[:, :, 2].mean(axis=1)
            # get the sorting indices (from farthest to nearest)
            depth_sorted_indices = np.argsort(faces_depth)
            # apply the sorting to faces_vertices_2d and faces_hidden
            faces_vertices_2d = faces_vertices_2d[depth_sorted_indices]
            faces_color = faces_color[depth_sorted_indices]

        # =============================================================================
        # honor material.face_culling
        # =============================================================================

        faces_visible = RendererUtils.compute_faces_visible(faces_vertices_2d, material.face_culling)
        # print(f"faces_visible: {faces_visible.sum()}/{len(faces_visible)}")

        # remove hidden faces - CAUTION: must be done after sorting be sure to keep the order of ALL arrays the same
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
        # Update all the artists
        # =============================================================================

        # update the PathCollection with the new patches
        mpl_poly_collection.set_verts(typing.cast(list, faces_vertices_2d))
        mpl_poly_collection.set_facecolor(typing.cast(list, faces_color))
        mpl_poly_collection.set_edgecolor(typing.cast(list, material.edge_colors))
        mpl_poly_collection.set_linewidth(typing.cast(list, material.edge_widths))

        return [mpl_poly_collection]
