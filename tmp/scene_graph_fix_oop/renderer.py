"""Minimal matplotlib renderer for the standalone OOP scene graph."""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
from pyrr import Matrix44, Vector3

from .camera import Camera
from .object3d import Object3D


class Renderer:
    """Project scene nodes into 2D and plot them as points."""

    def render(self, camera: Camera, scene: Object3D) -> None:
        scene.update_world_transform()
        camera.update_world_transform()

        view = Matrix44(camera.get_view_matrix(), dtype=np.float32)
        projection = Matrix44(camera.get_projection_matrix(), dtype=np.float32)

        xs: List[float] = []
        ys: List[float] = []

        for node in self._traverse(scene):
            if node is scene or node is camera:
                continue

            world_pos = Vector3(node.get_world_position(), dtype=np.float32)
            homogenous = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0], dtype=np.float32)

            camera_space = view @ homogenous
            clip_space = projection @ camera_space

            w = clip_space[3]
            if np.isclose(w, 0.0):
                continue

            ndc = clip_space[:3] / w

            if -1.0 <= ndc[0] <= 1.0 and -1.0 <= ndc[1] <= 1.0 and -1.0 <= ndc[2] <= 1.0:
                xs.append(float(ndc[0]))
                ys.append(float(ndc[1]))

        plt.figure(figsize=(6, 6))
        plt.scatter(xs, ys, c="blue", s=30)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scene Graph Projection")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()

    def _traverse(self, root: Object3D) -> List[Object3D]:
        stack = [root]
        ordered: List[Object3D] = []
        while stack:
            node = stack.pop()
            ordered.append(node)
            stack.extend(node.children)
        return ordered
