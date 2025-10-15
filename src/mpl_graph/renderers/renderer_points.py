# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections
import numpy as np

# local imports
from ..objects.points import Points
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..math.transform_utils import TransformUtils
from ..geometry.geometry_utils import GeometryUtils
from ..renderers.renderer_utils import RendererUtils


class RendererPoints:
    @staticmethod
    def render(renderer: "Renderer", points: Points, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = points.geometry
        material = points.material

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        # Get the vertices in local space
        vertices_localspace = geometry.vertices

        # full_transform = points.get_world_matrix()
        mvp_matrix = TransformUtils.compute_mvp_matrix(camera, points)
        vertices_npc, vertices_clip = GeometryUtils.apply_mvp_matrix(vertices_localspace, mvp_matrix)

        # dispatch the post_transforming event
        points.post_transform.dispatch(vertices_clip)

        # =============================================================================
        # Depth sort at the faces level
        # =============================================================================

        # Sort polygons by depth (painter's algorithm)
        if material.depth_sorting:
            # compute the depth of each face as the mean z value of its vertices
            points_depth = vertices_npc[:, 2]
            # get the sorting indices (from farthest to nearest)
            depth_sorted_indices = np.argsort(points_depth)
            # apply the sorting to vertices_npc
            vertices_npc = vertices_npc[depth_sorted_indices]

        # =============================================================================
        # Switch vertices to 2d
        # =============================================================================

        # drop z for 2D rendering
        vertices_2d = vertices_npc[:, :2]

        # =============================================================================
        # Create the artists if needed
        # =============================================================================
        if points.uuid not in renderer._artists:
            mpl_path_collection = renderer._axis.scatter([], [])  # type: ignore
            mpl_path_collection.set_visible(False)  # hide until properly positioned and sized
            renderer._axis.add_collection(mpl_path_collection)
            renderer._artists[points.uuid] = mpl_path_collection

        # =============================================================================
        # Get the mpl_artist
        # =============================================================================

        mpl_path_collection = typing.cast(matplotlib.collections.PathCollection, renderer._artists[points.uuid])
        mpl_path_collection.set_visible(True)

        # =============================================================================
        # do z-ordering based on distance to camera
        # =============================================================================

        # compute and set zorder on our single artist
        RendererUtils.update_single_artist_zorder(camera, points, mpl_path_collection)

        # =============================================================================
        # Update the matplotlib artists data
        # =============================================================================

        mpl_path_collection.set_offsets(offsets=vertices_2d)
        mpl_path_collection.set_sizes(typing.cast(list, material.sizes))  # set a default size for each point
        mpl_path_collection.set_color(typing.cast(list, material.colors))
        mpl_path_collection.set_edgecolor(typing.cast(list, material.edge_colors))
        mpl_path_collection.set_linewidth(typing.cast(list, material.edge_widths))

        return [mpl_path_collection]
