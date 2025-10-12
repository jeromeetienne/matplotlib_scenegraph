# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections

# local imports
from ..objects.points import Points
from ..renderers.renderer import Renderer
from ..cameras.camera_base import CameraBase
from ..core.transform_utils import TransformUtils


class RendererPoints:
    @staticmethod
    def render(renderer: "Renderer", points: Points, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        geometry = points.geometry
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
        # Get the existing mpl artist
        mpl_path_collection = typing.cast(matplotlib.collections.PathCollection, renderer._artists[points.uuid])
        # set it to visible
        mpl_path_collection.set_visible(True)

        # =============================================================================
        # Apply full transform the vertices
        # =============================================================================

        full_transform = points.get_world_matrix()
        # full_transform = TransformUtils.compute_full_transform(camera, points)
        vertices_transformed = TransformUtils.apply_transform(geometry.vertices, full_transform)

        # dispatch the post_transforming event
        points.post_transform.dispatch(renderer=renderer, camera=camera, vertices_transformed=vertices_transformed)

        vertices_2d = vertices_transformed[:, :2]  # drop z for 2D rendering

        # =============================================================================
        # Update the matplotlib artists data
        # =============================================================================

        mpl_path_collection.set_offsets(offsets=vertices_2d)
        mpl_path_collection.set_sizes(points.sizes.tolist())  # set a default size for each point
        mpl_path_collection.set_color(points.colors.tolist())
        mpl_path_collection.set_edgecolor(points.edge_colors.tolist())
        mpl_path_collection.set_linewidth(points.edge_widths.tolist())

        return [mpl_path_collection]
