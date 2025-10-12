# stdlib imports
from typing import Callable, Sequence

# pip imports
import time
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot

# local imports
from mpl_graph.core.object_3d import Object3D
from mpl_graph.cameras.camera_base import CameraBase
from mpl_graph.renderers.renderer import RendererMatplotlib

# do a callback type for the animation loop
AnimationLoopCallbackType = Callable[[float, float], Sequence[Object3D]]
"""A simple animation loop manager for matplotlib rendering.

Arguments:
    delta_time (float): Time elapsed since the last frame in milliseconds.
    timestamp (float): Total time elapsed since the start of the animation in milliseconds.
"""


class AnimationLoop:
    def __init__(self, renderer: RendererMatplotlib, fps: int = 30, video_duration: float = 10.0, video_path: str | None = None) -> None:
        self._callbacks = []
        self._renderer = renderer
        self._fps = fps
        self._video_duration = video_duration
        self._video_path = video_path

    def start(self, scene: Object3D, camera: CameraBase):
        time_start = time.time()
        time_last = time_start

        # initial render
        self._renderer.render(scene, camera)

        # define a animation function for matplotlib
        def update_scene(frame) -> list[matplotlib.artist.Artist]:
            nonlocal time_last, time_start
            present = time.time()
            timestamp = present - time_start
            delta_time = timestamp - time_last
            time_last = present

            changed_objects: list[Object3D] = []
            for callback in self._callbacks:
                _changed_objects = callback(delta_time, timestamp)
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
            self._renderer.get_figure(), update_scene, frames=int(self._video_duration * self._fps), interval=1000 / self._fps
        )
        if self._video_path is not None:
            funcAnimation.save(self._video_path, dpi=200, fps=self._fps)

        matplotlib.pyplot.show(block=True)

    def stop(self):
        raise NotImplementedError()

    def add_callback(self, func: AnimationLoopCallbackType):
        self._callbacks.append(func)

    def remove_callback(self, func: AnimationLoopCallbackType):
        self._callbacks.remove(func)
