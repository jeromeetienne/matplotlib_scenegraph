# stdlib imports
import typing

# pip imports
import matplotlib.pyplot
import matplotlib.artist
import matplotlib.figure
import matplotlib.axes
import numpy as np

# local imports
from ..core.object_3d import Object3D
from ..core import Constants
from ..objects.points import Points
from ..objects.lines import Lines
from ..objects.polygons import Polygons
from ..objects.sprite import Sprite
from ..objects.mesh import Mesh
from ..objects.scene import Scene
from ..objects.text import Text
from ..cameras.camera import Camera


class Renderer:
    __slot__ = "depth_sorting"

    def __init__(
        self, figure_w: int = 256, figure_h: int = 256, dpi: int = 100, /, depth_sorting: bool = False, background_color: np.ndarray | None = None
    ) -> None:
        self.width = figure_w
        """Width of the figure in pixels."""
        self.height = figure_h
        """Height of the figure in pixels."""
        self.dpi = dpi
        """DPI (dots per inch) of the figure."""
        self.background_color = background_color if background_color is not None else Constants.Color.WHITE
        """Background color of the figure."""

        # =============================================================================
        # Setup matplotlib
        # =============================================================================

        # Create a figure of 512x512 pixels
        self._figure = matplotlib.pyplot.figure(figsize=(self.width / self.dpi, self.height / self.dpi), dpi=self.dpi)

        # Set the figure's background color
        self._figure.patch.set_facecolor(self.background_color.tolist())

        # Create an axis that fills the whole figure
        self._axis = self._figure.add_axes((0, 0, 1, 1), frameon=False)

        # this should be -1 to 1 - from normalized device coordinates - https://en.wikipedia.org/wiki/Graphics_pipeline
        # - https://developer.mozilla.org/en-US/docs/Web/API/WebGL_API/WebGL_model_view_projection
        self._axis.set_xlim(-1, 1)
        self._axis.set_ylim(-1, 1)
        self._artists: dict[str, matplotlib.artist.Artist] = {}

    def close(self) -> None:
        # stop the event loop if any - thus .show(block=True) will return
        self._figure.canvas.stop_event_loop()
        # close the figure
        matplotlib.pyplot.close(self._figure)

    # =============================================================================
    # Public Function
    # =============================================================================

    def get_figure(self) -> matplotlib.figure.Figure:
        return self._figure

    def get_axis(self) -> matplotlib.axes.Axes:
        return self._axis

    def render(self, scene: Scene, camera: Camera) -> list[matplotlib.artist.Artist]:
        # update world matrices
        scene.update_world_matrix()

        # render objects
        changed_artists: list[matplotlib.artist.Artist] = []
        for object3d in scene.traverse():
            _changed_artists = self._render_object(object3d, camera)
            changed_artists.extend(_changed_artists)

        return changed_artists

    def render_object(self, object3d: Object3D, camera: Camera) -> list[matplotlib.artist.Artist]:
        changed_artists: list[matplotlib.artist.Artist] = self._render_object(object3d, camera)
        return changed_artists

    # =============================================================================
    # Private functions
    # =============================================================================
    def _render_object(self, object3d: Object3D, camera: Camera) -> list[matplotlib.artist.Artist]:

        # =============================================================================
        # Dispatch pre_rendering Event
        # =============================================================================

        # dispatch the pre_rendering event
        object3d.pre_rendering.dispatch(renderer=self, camera=camera)

        # =============================================================================
        # Render the object based on its type
        # =============================================================================
        changed_artists: list[matplotlib.artist.Artist] = []

        # call the appropriate renderer based on the object type
        if isinstance(object3d, Points):
            from .renderer_points import RendererPoints

            _changed_artists = RendererPoints.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Lines):
            from .renderer_lines import RendererLines

            _changed_artists = RendererLines.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Mesh):
            from .renderer_mesh import RendererMesh

            _changed_artists = RendererMesh.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Polygons):
            from .renderer_polygons import RendererPolygons

            _changed_artists = RendererPolygons.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Sprite):
            from .renderer_sprite import RendererSprite

            _changed_artists = RendererSprite.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Text):
            from .renderer_text import RendererText

            _changed_artists = RendererText.render(self, object3d, camera)
            changed_artists.extend(_changed_artists)
        elif isinstance(object3d, Object3D):
            # base class, do nothing
            pass
        else:
            raise NotImplementedError(f"Rendering for {type(object3d)} not implemented yet")

        # =============================================================================
        # Dispatch post_rendering Event
        # =============================================================================

        # dispatch the post_rendering event
        object3d.post_rendering.dispatch(renderer=self, camera=camera)

        # return the list of changed artists
        return changed_artists
