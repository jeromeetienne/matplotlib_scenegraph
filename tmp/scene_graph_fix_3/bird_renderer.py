"""Quick matplotlib-based renderer for visualising a scene graph from above."""

from __future__ import annotations

import math
import time
from typing import Iterable, List

import matplotlib.pyplot as plt
import matplotlib.axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # ensures 3D backend is registered

from object3d import Object3D
from perpective_camera import PerspectiveCamera


def _iter_scene(root: Object3D) -> Iterable[Object3D]:
    stack: List[Object3D] = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _world_position(node: Object3D) -> np.ndarray:
    origin = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return origin @ node.world_transform


def _world_forward(node: Object3D) -> np.ndarray:
    forward = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    transformed = forward @ node.world_transform
    direction = transformed[:3]
    length = np.linalg.norm(direction)
    if length == 0.0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return direction / length


class BirdRenderer:
    """Render the scene graph from a bird's-eye view using matplotlib 3D."""

    def __init__(self, figsize: tuple[int, int] = (8, 6)) -> None:
        self.figsize = figsize

    def render(self, scene_root: Object3D, camera: PerspectiveCamera | None = None, animate: bool = False) -> None:
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=60.0, azim=-45.0)

        def update(frame: int):
            nonlocal ax, scene_root
            timestamp = time.time()
            angle = math.radians(frame * 2.0)
            object3d = scene_root.get_by_name("root")
            if object3d is not None:
                # child.position[1] = math.sin(angle) * 2
                object3d.scale[:] = 1.0 + 0.5 * math.sin(timestamp)
                # child.scale = np.array([1.0 + 0.2 * math.sin(angle), 1.0, 1.0 + 0.2 * math.cos(angle)], dtype=np.float64)
                # child.rotate_y(math.radians(4.0))

            scene_root.update_world_transform()
            BirdRenderer.draw_scene(ax, scene_root)
            return []

        scene_root.update_world_transform()

        if animate:
            anim = animation.FuncAnimation(fig, update, frames=180, interval=50, repeat=True)
            BirdRenderer.draw_scene(ax, scene_root)
            plt.show()
        else:
            BirdRenderer.draw_scene(ax, scene_root)
            plt.show()

    @staticmethod
    def draw_scene(ax: Axes3D, scene_root: Object3D) -> None:
        elev, azim = ax.elev, ax.azim
        ax.clear()
        ax.view_init(elev=elev, azim=azim)

        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []

        max_extent = 1.0

        for node in scene_root.traverse():
            pos = _world_position(node)
            forward = _world_forward(node)
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])

            line_end = pos[:3] + forward * 0.5
            ax.plot([pos[0], line_end[0]], [pos[1], line_end[1]], [pos[2], line_end[2]], color="crimson")

            if node.parent is not None:
                parent_pos = _world_position(node.parent)
                ax.plot(
                    [parent_pos[0], pos[0]],
                    [parent_pos[1], pos[1]],
                    [parent_pos[2], pos[2]],
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                )

            label = node.name if node.name is not None else "(unnamed)"
            ax.text(pos[0], pos[1], pos[2], label, fontsize=9, color="black")

            max_extent = max(max_extent, np.linalg.norm(pos[:3]))

        if xs:
            ax.scatter(xs, ys, zs, color="royalblue", s=40)  # type: ignore[arg-type]

        extent = max_extent + 1.0
        extent = 5.0
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Bird's-eye Scene Overview")
        ax.grid(True)

        plt.tight_layout()


# =============================================================================
# make demo scene
# =============================================================================


def _make_demo_scene() -> tuple[Object3D, PerspectiveCamera]:
    root = Object3D(name="root")

    # Create a few child nodes with varying transforms.
    child = Object3D(name="child")
    child.position = np.array([1.5, 0.0, 0.0], dtype=np.float64)
    child.rotate_y(math.radians(35.0))

    grandchild = Object3D(name="grandchild")
    grandchild.position = np.array([0.0, 0.5, 1.5], dtype=np.float64)
    grandchild.rotate_x(math.radians(-20.0))

    grandgrandchild = Object3D(name="grandgrandchild")
    grandgrandchild.position = np.array([0.0, 0.2, 0.6], dtype=np.float64)

    wing_left = Object3D(name="wing_L")
    wing_left.position = np.array([-1.5, 0.0, 0.5], dtype=np.float64)

    wing_right = Object3D(name="wing_R")
    wing_right.position = np.array([-1.5, 0.0, -0.5], dtype=np.float64)

    root.add_child(child)
    child.add_child(grandchild)
    grandchild.add_child(grandgrandchild)
    root.add_child(wing_left)
    root.add_child(wing_right)

    camera = PerspectiveCamera(fov=60.0)
    camera.position = np.array([0.0, 5.0, 8.0], dtype=np.float64)
    camera.rotate_x(math.radians(-30.0))
    camera.update_world_transform()

    return root, camera


if __name__ == "__main__":
    scene, camera = _make_demo_scene()
    renderer = BirdRenderer()
    renderer.render(scene, camera, animate=True)
