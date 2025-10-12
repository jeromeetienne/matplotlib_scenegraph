# pip imports
import matplotlib.pyplot

# local imports
from mpl_graph.renderers.renderer import RendererMatplotlib
from mpl_graph.core.object_3d import Object3D

class CameraController:
    def __init__(self, renderer: RendererMatplotlib, scene: Object3D, camera: Object3D):
        self._renderer = renderer
        self._camera = camera
        self._scene = scene
        matplotlib.pyplot.rcParams['keymap.save'].remove('s')
        self._mpl_connect_id = None

    def start(self):
        self._mpl_connect_id = self._renderer.get_figure().canvas.mpl_connect('key_press_event', self._on_key)

    def stop(self):
        self._renderer.get_figure().canvas.mpl_disconnect('key_press_event', self._mpl_connect_id)
        self._mpl_connect_id = None

    def _on_key(self, event):
        print(f"Key pressed: {event.key}")
        speed = 0.1
        if (event.key == 'w' or event.key == 'up') or (event.key == 'shift+w' or event.key == 'shift+up'):
            self._camera.position[1] += speed
        elif (event.key == 's' or event.key == 'down') or (event.key == 'shift+s' or event.key == 'shift+down'):
            self._camera.position[1] -= speed       
        elif event.key == 'shift+a' or event.key == 'shift+left':
            self._camera.position[0] -= speed
        elif event.key == 'shift+d' or event.key == 'shift+right':
            self._camera.position[0] += speed
        elif event.key == 'a' or event.key == 'left':
            self._camera.rotation_euler[2] += 0.1
        elif event.key == 'd' or event.key == 'right':
            self._camera.rotation_euler[2] -= 0.1

        print(f"Camera position: {self._camera.get_world_position()} rotation: {self._camera.get_world_rotation_euler()}")

        # re-render the scene
        self._renderer.render(self._scene, self._camera)
