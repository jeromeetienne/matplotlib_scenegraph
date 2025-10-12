"""Minimal matplotlib renderer for the scene graph."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pyrr import matrix44

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from camera import Camera
from object3d import Object3D


class Renderer:
    """Render scene nodes as projected points."""

    def __init__(self) -> None:
        # Matplotlib figure and axes for rendering
        self._figure: Optional[Figure] = None
        self._axes: Optional[Axes] = None

    def render(self, camera: Camera, scene: Object3D) -> Axes:
        """
        Render the scene from the perspective of the given camera.

        Args:
            camera: The camera object used for view/projection.
            scene: The root Object3D node of the scene graph.

        Returns:
            The matplotlib Axes object with the rendered scene.
        """
        # Update transforms for all nodes
        scene.update_world_transform()
        camera.update_world_transform()
        # Get camera view and projection matrices
        view_matrix = camera.get_view_matrix()
        projection_matrix = camera.get_projection_matrix()
        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []
        # Traverse all nodes in the scene graph
        for node in scene.traverse():
            # if node is scene or node is camera:
            #     continue  # Skip root scene and camera nodes
            # Get world position of node
            position = node.world_position()
            # Convert to homogeneous coordinates
            position4 = np.array([position[0], position[1], position[2], 1.0], dtype=float)
            # Transform to view space
            view_space = matrix44.apply_to_vector(view_matrix, position4)
            # Transform to clip space
            clip_space = matrix44.apply_to_vector(projection_matrix, view_space)
            w_component = clip_space[3]
            if w_component == 0.0:
                continue  # Avoid division by zero
            # Perspective divide to get normalized device coordinates (NDC)
            ndc = clip_space[:3] / w_component
            # Cull points outside the NDC cube
            if np.any(np.abs(ndc[:2]) > 1.0) or ndc[2] < -1.0 or ndc[2] > 1.0:
                continue
            xs.append(ndc[0])
            ys.append(ndc[1])
            labels.append(node.name)
        # Create figure/axes if not already created
        if self._figure is None or self._axes is None:
            self._figure, self._axes = plt.subplots(figsize=(6, 6))
        else:
            self._axes.clear()
        # Scatter plot for projected points
        self._axes.scatter(xs, ys, c="tab:blue")
        # Draw labels for each point
        for x, y, label in zip(xs, ys, labels):
            self._axes.text(x, y, label, fontsize=8, ha="center")
        # Set axes limits and labels
        self._axes.set_xlim(-1.1, 1.1)
        self._axes.set_ylim(-1.1, 1.1)
        self._axes.set_xlabel("X (NDC)")
        self._axes.set_ylabel("Y (NDC)")
        self._axes.set_title("Scene Graph View")
        self._axes.grid(True, linestyle="--", alpha=0.4)
        self._axes.set_aspect("equal")
        # Redraw the canvas
        self._figure.canvas.draw_idle()
        plt.pause(0.001)
        return self._axes
