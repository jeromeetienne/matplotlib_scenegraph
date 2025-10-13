# pip imports
import matplotlib.pyplot
import matplotlib.artist
import matplotlib.figure
import matplotlib.axes

# local imports
from ..core.object_3d import Object3D
from ..objects.points import Points
from ..objects.lines import Lines
from ..objects.polygons import Polygons
from ..objects.sprite import Sprite
from ..objects.textured_mesh import TexturedMesh
from ..cameras.camera_base import CameraBase


class Renderer:
    __slot__ = "depth_sorting"

    def __init__(self, figure_w: int = 100, figure_h: int = 100, dpi: int = 100, /, depth_sorting: bool = True) -> None:
        # Create a figure of 512x512 pixels
        self._figure = matplotlib.pyplot.figure(figsize=(figure_w / dpi, figure_h / dpi), dpi=dpi)

        self.depth_sorting = depth_sorting
        """Whether to enable depth sorting based on camera distance at the object3D level.
        This affects the `zorder` of the matplotlib artists created for each object3D."""

        # Create an axis that fills the whole figure
        self._axis = self._figure.add_axes((0, 0, 1, 1), frameon=False)

        # this should be -1 to 1
        # for perspective camera - BUG BUG
        self._axis.set_xlim(-3, 3)
        self._axis.set_ylim(-3, 3)
        # for orthographic camera - BUG BUG
        self._axis.set_xlim(-1, 1)
        self._axis.set_ylim(-1, 1)
        self._artists: dict[str, matplotlib.artist.Artist] = {}

    # =============================================================================
    # Public Function
    # =============================================================================

    def get_figure(self) -> matplotlib.figure.Figure:
        return self._figure

    def get_axis(self) -> matplotlib.axes.Axes:
        return self._axis

    def render(self, scene: Object3D, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        # update world matrices
        scene.update_world_matrix()

        # render objects
        changed_artists: list[matplotlib.artist.Artist] = []
        for object3d in scene.traverse():
            _changed_artists = self._render_object(object3d, camera)
            changed_artists.extend(_changed_artists)

        return changed_artists

    def render_object(self, object3d: Object3D, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        changed_artists: list[matplotlib.artist.Artist] = self._render_object(object3d, camera)
        return changed_artists

    # =============================================================================
    # Private functions
    # =============================================================================
    def _render_object(self, object3d: Object3D, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        changed_artists: list[matplotlib.artist.Artist] = []

        # =============================================================================
        # Dispatch pre_rendering Event
        # =============================================================================

        # dispatch the pre_rendering event
        object3d.pre_rendering.dispatch(renderer=self, camera=camera)

        # =============================================================================
        # Render the object based on its type
        # =============================================================================

        # call the appropriate renderer based on the object type
        if isinstance(object3d, Points):
            from .renderer_points import RendererPoints

            _changed_artists = RendererPoints.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Lines):
            from .renderer_lines import RendererLines

            _changed_artists = RendererLines.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Polygons):
            from .renderer_polygons import RendererPolygons

            _changed_artists = RendererPolygons.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Sprite):
            from .renderer_sprite import RendererSprite

            _changed_artists = RendererSprite.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, TexturedMesh):
            from .renderer_textured_mesh import RendererTexturedMesh

            _changed_artists = RendererTexturedMesh.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Object3D):
            # base class, do nothing
            pass
        else:
            raise NotImplementedError(f"Rendering for {type(object3d)} not implemented yet")

        # =============================================================================
        # honor .depth_sorting
        # =============================================================================

        # update `artists.zorder` based on camera distance if depth sorting is enabled
        if self.depth_sorting:
            # compute distance from camera to object3d
            camera_position = camera.get_world_position()
            object_position = object3d.get_world_position()
            euclidian_distance = ((camera_position - object_position) ** 2).sum() ** 0.5
            # set zorder based on distance (larger distance -> smaller zorder)
            zorder = -euclidian_distance
            for artist in changed_artists:
                artist.set_zorder(zorder)

        # =============================================================================
        # Dispatch post_rendering Event
        # =============================================================================

        # dispatch the post_rendering event
        object3d.post_rendering.dispatch(renderer=self, camera=camera)

        # return the list of changed artists
        return changed_artists
