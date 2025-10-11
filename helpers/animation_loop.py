import typing
import time
import matplotlib.animation
import matplotlib.artist
import matplotlib.pyplot
from core.object_3d import Object3D
from cameras.camera_base import CameraBase
from renderers.matplotlib.renderer import RendererMatplotlib

# do a callback type for the animation loop
AnimationLoopCallbackType = typing.Callable[[float, float], list[Object3D]]
"""A simple animation loop manager for matplotlib rendering.

Arguments:
    delta_time (float): Time elapsed since the last frame in milliseconds.
    timestamp (float): Total time elapsed since the start of the animation in milliseconds.
"""


class AnimationLoop:
    def __init__(self, renderer: RendererMatplotlib):
        self._callbacks = []
        self._renderer = renderer

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

            changed_artists: list[matplotlib.artist.Artist] = []
            for object in changed_objects:
                _changed_artists = self._renderer.render_object(object, camera)
                changed_artists.extend(_changed_artists)

            # print(f"  Number of changed artists: {len(changed_artists)}")
            return changed_artists

        ani = matplotlib.animation.FuncAnimation(self._renderer._figure, update_scene, frames=100, interval=1000 / 60)

        matplotlib.pyplot.show()

    def stop(self):
        raise NotImplementedError()

    def add_callback(self, func: AnimationLoopCallbackType):
        self._callbacks.append(func)

    def remove_callback(self, func: AnimationLoopCallbackType):
        self._callbacks.remove(func)
