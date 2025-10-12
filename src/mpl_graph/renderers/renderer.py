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
    def __init__(self, figure_w: int = 100, figure_h: int = 100, dpi: int = 100):
        # Create a figure of 512x512 pixels
        self._figure = matplotlib.pyplot.figure(figsize=(figure_w / dpi, figure_h / dpi), dpi=dpi)

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

    def get_figure(self) -> matplotlib.figure.Figure:
        return self._figure

    def get_axis(self) -> matplotlib.axes.Axes:
        return self._axis

    def render(self, scene: Object3D, camera: CameraBase) -> list[matplotlib.artist.Artist]:
        # update world matrices
        scene.update_world_matrix()

        # render from back to front
        # (not implemented here, just a placeholder comment)

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

        # dispatch the pre_rendering event
        object3d.pre_rendering.dispatch(renderer=self, camera=camera)

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

        # dispatch the post_rendering event
        object3d.post_rendering.dispatch(renderer=self, camera=camera)

        # return the list of changed artists
        return changed_artists
