# stdlib imports
from typing import Callable, Sequence
from typing import Protocol

# pip imports
import time
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot

# local imports
from common.example_utils import ExamplesUtils
from mpl_graph.core.object_3d import Object3D
from mpl_graph.objects import Scene
from mpl_graph.core.event import Event
from mpl_graph.cameras.camera_base import CameraBase
from mpl_graph.renderers.renderer import Renderer

# do a callback type for the animation loop
AnimationLoopCallbackType = Callable[[float], Sequence[Object3D]]
"""A simple animation loop manager for matplotlib rendering.

Arguments:
    delta_time (float): Time elapsed since the last frame in milliseconds.
"""


class VideoSavedCalledback(Protocol):
    def __call__(self) -> None: ...  # type: ignore


class AnimationLoop:
    __slots__ = ("_callbacks", "_renderer", "_fps", "_video_duration", "_video_path", "_time_last_update", "_scene", "_camera", "video_saved", "_funcAnimation")

    def __init__(self, renderer: Renderer, fps: int = 30, video_duration: float = 10.0, video_path: str | None = None) -> None:
        """
        A simple animation loop manager for matplotlib rendering.
        - it is able to save a video if needed
        - it calls registered callbacks to update the scene
        - it re-renders only the changed objects

        Arguments:
            renderer (Renderer): The renderer to use for rendering the scene.
            fps (int): The target frames per second for the animation loop.
            video_duration (float): The duration of the video to save in seconds.
            video_path (str | None): The path to save the video. If None, no video is saved.
        """
        self._callbacks: list[AnimationLoopCallbackType] = []
        self._renderer = renderer
        self._fps = fps
        self._video_duration = video_duration
        self._video_path = video_path
        self._time_last_update = None
        self._scene: Object3D | None = None
        self._camera: CameraBase | None = None
        self._funcAnimation: matplotlib.animation.FuncAnimation | None = None

        self.video_saved = Event[VideoSavedCalledback]()
        """Event triggered when the video is saved."""

    # =============================================================================
    # .start/.stop
    # =============================================================================

    def start(self, scene: Scene, camera: CameraBase):
        """Start the animation loop."""

        self._scene = scene
        self._camera = camera
        self._time_last_update = time.time()

        # initial render
        self._renderer.render(scene, camera)

        # if we are in testing mode, exit now
        if ExamplesUtils.postamble():
            return

        self._funcAnimation = matplotlib.animation.FuncAnimation(
            self._renderer.get_figure(), self._mpl_update_scene, frames=int(self._video_duration * self._fps), interval=1000 / self._fps
        )
        if self._video_path is not None:
            self._funcAnimation.save(self._video_path, dpi=200, fps=self._fps)
            print(f"Video saved to: {self._video_path}")
            self.video_saved.dispatch()

        matplotlib.pyplot.show(block=True)

    def stop(self):
        self._scene = None
        self._camera = None
        self._time_last_update = None

        # stop the animation function timer
        if self._funcAnimation is not None:
            self._funcAnimation.event_source.stop()
            self._funcAnimation = None

    # =============================================================================
    # .add_callback/.remove_callback/.decorator
    # =============================================================================

    def add_callback(self, func: AnimationLoopCallbackType):
        """Add a callback to the animation loop."""
        self._callbacks.append(func)

    def remove_callback(self, func: AnimationLoopCallbackType):
        """Remove a callback from the animation loop."""
        self._callbacks.remove(func)

    def callback_decorator(self, func: AnimationLoopCallbackType) -> AnimationLoopCallbackType:
        """A decorator to add a callback to the animation loop.

        Usage:
            ```python
                @animation_loop.decorator
                def my_callback(delta_time: float) -> Sequence[Object3D]:
                    ...

                # later, if needed
                animation_loop.remove_callback(my_callback)
            ```
        """

        self.add_callback(func)

        def wrapper(delta_time: float) -> Sequence[Object3D]:
            # print("Before the function runs")
            result = func(delta_time)
            # print("After the function runs")
            return result

        return wrapper

    # define a animation function for matplotlib
    def _mpl_update_scene(self, frame) -> list[matplotlib.artist.Artist]:
        # sanity checks
        assert self._scene is not None, "Scene is not set"
        assert self._camera is not None, "Camera is not set"

        # compute delta time
        present = time.time()
        delta_time = (present - self._time_last_update) if self._time_last_update is not None else (1 / self._fps)
        self._time_last_update = present

        # notify all callbacks
        changed_objects: list[Object3D] = []
        for callback in self._callbacks:
            _changed_objects = callback(delta_time)
            changed_objects.extend(_changed_objects)

        # update world matrices
        self._scene.update_world_matrix()

        # render only the changed objects
        changed_artists: list[matplotlib.artist.Artist] = []
        for object in changed_objects:
            # needed to update the attributes of the object3D, like position, rotation, scale, etc.
            _changed_artists = self._renderer.render_object(object, self._camera)
            changed_artists.extend(_changed_artists)

        # print(f"  Number of changed artists: {len(changed_artists)}")
        return changed_artists
