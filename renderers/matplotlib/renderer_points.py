# stdlib imports
import typing

# pip imports
import matplotlib.artist
import matplotlib.collections

# local imports
from objects.points import Points
from renderers.matplotlib.renderer import RendererMatplotlib
from cameras.camera_base import CameraBase
from core.transform_utils import TransformUtils


class MatplotlibRendererPoints:
    @staticmethod
    def render(renderer: "RendererMatplotlib", points: Points, camera: CameraBase) -> list[matplotlib.artist.Artist]:

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

        # full_transform = points.get_world_matrix()
        full_transform = TransformUtils.compute_full_transform(camera, points)
        vertices = TransformUtils.apply_transform(points.vertices, full_transform)

        vertices_2d = vertices[:, :2]  # drop z for 2D rendering

        # =============================================================================
        # Update the matplotlib artists data
        # =============================================================================

        mpl_path_collection.set_offsets(offsets=vertices_2d)
        mpl_path_collection.set_sizes([1] * len(points.vertices))  # set a default size for each point
        mpl_path_collection.set_color(points.color.tolist())
        # mpl_path_collection.set_edgecolor((0, 0, 0, 1))
        # mpl_path_collection.set_linewidth(2)

        return [mpl_path_collection]
