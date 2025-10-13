# stdlib imports
from typing import Callable, Sequence

# pip imports
import time
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot

# local imports
from common.example_utils import ExamplesUtils
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_base import CameraBase
from mpl_graph.renderers.renderer import Renderer

# do a callback type for the animation loop
AnimationLoopCallbackType = Callable[[float], Sequence[Object3D]]
"""A simple animation loop manager for matplotlib rendering.

Arguments:
    delta_time (float): Time elapsed since the last frame in milliseconds.
"""


class AnimationLoop:
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

    # =============================================================================
    # .start/.stop
    # =============================================================================

    def start(self, scene: Object3D, camera: CameraBase):
        """Start the animation loop."""
        time_start = time.time()
        time_last = time_start

        # initial render
        self._renderer.render(scene, camera)

        # if we are in testing mode, exit now
        if ExamplesUtils.postamble():
            return

        # define a animation function for matplotlib
        def mpl_update_scene(frame) -> list[matplotlib.artist.Artist]:
            # TODO put this function non local, self._mpl_update_scene
            nonlocal time_last, time_start
            present = time.time()
            timestamp = present - time_start
            delta_time = present - time_last
            time_last = present
            # print(f"AnimationLoop.mpl_update_scene: frame={frame}, timestamp={timestamp}, delta_time={delta_time}")

            changed_objects: list[Object3D] = []
            for callback in self._callbacks:
                _changed_objects = callback(delta_time)
                changed_objects.extend(_changed_objects)

            # update world matrices
            scene.update_world_matrix()

            changed_artists: list[matplotlib.artist.Artist] = []
            for object in changed_objects:
                _changed_artists = self._renderer.render_object(object, camera)
                changed_artists.extend(_changed_artists)

            # print(f"  Number of changed artists: {len(changed_artists)}")
            return changed_artists

        funcAnimation = matplotlib.animation.FuncAnimation(
            self._renderer.get_figure(), mpl_update_scene, frames=int(self._video_duration * self._fps), interval=1000 / self._fps
        )
        if self._video_path is not None:
            funcAnimation.save(self._video_path, dpi=200, fps=self._fps)

        matplotlib.pyplot.show(block=True)

    def stop(self):
        """Stop the animation loop."""
        raise NotImplementedError()

    # =============================================================================
    # .add_callback/.remove_callback/.decorator
    # =============================================================================

    def add_callback(self, func: AnimationLoopCallbackType):
        """Add a callback to the animation loop."""
        self._callbacks.append(func)

    def remove_callback(self, func: AnimationLoopCallbackType):
        """Remove a callback from the animation loop."""
        self._callbacks.remove(func)

    def decorator(self, func: AnimationLoopCallbackType) -> AnimationLoopCallbackType:
        """A decorator to add a callback to the animation loop. NOTE: this callback will never be removed.

        Usage:
            ```python
                @animation_loop.decorator
                def my_callback(delta_time: float) -> Sequence[Object3D]:
                    ...
            ```
        """

        self.add_callback(func)

        def wrapper(delta_time: float) -> Sequence[Object3D]:
            # print("Before the function runs")
            result = func(delta_time)
            # print("After the function runs")
            return result

        return wrapper
