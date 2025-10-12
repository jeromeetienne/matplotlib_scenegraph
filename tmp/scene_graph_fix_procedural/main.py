"""Minimal entry point creating a demo scene and rendering it."""

from __future__ import annotations

import math
import time

import matplotlib.pyplot as plt
import numpy as np

from camera import Camera
from object3d import Object3D
from renderer import Renderer


def build_scene() -> tuple[Object3D, Camera]:
    """Builds a simple scene graph with a camera, a cube, and a child sphere."""
    scene = Object3D(name="Root")  # Root node of the scene graph
    # scene.scale[:] = 0.1

    camera = Camera(name="Camera", position=np.array([0.0, 0.0, 5.0], dtype=float))
    scene.add_child(camera)  # Add camera as a child of the root

    cube = Object3D(name="Cube")
    cube.position = np.array([-2.0, 0.0, 0.0], dtype=float)  # Set cube position
    # cube.scale = np.array([4.0, 4.0, 4.0], dtype=float)  # Set cube scale
    # cube.rotation_z(math.pi / 4.0)  # Rotate cube 45 degrees around Y axis
    scene.add_child(cube)  # Add cube to the scene

    sphere = Object3D(name="Sphere")
    sphere.position = np.array([0.0, 1.0, 0.0], dtype=float)  # Set sphere position
    # sphere.scale = np.array([0.5, 0.5, 0.5], dtype=float)  # Scale sphere down
    cube.add_child(sphere)  # Add sphere as a child of the cube

    triangle = Object3D(name="Triangle")
    triangle.position = np.array([-0.5, 0.0, 0.0], dtype=float)  # Set triangle position
    # triangle.scale = np.array([0.5, 0.5, 0.5], dtype=float)  # Scale triangle down
    sphere.add_child(triangle)  # Add triangle as a child of the sphere

    return scene, camera


def main() -> None:
    """Creates the scene, renders it, and displays the result."""
    scene, camera = build_scene()
    renderer = Renderer()
    renderer.render(camera, scene)
    # plt.show()  # Display the rendered scene

    while True:
        print(".", end="", flush=True)
        timestamp = time.time()
        object = scene.get_object_by_name("Sphere")
        assert object is not None
        # cube.rotation_y(0.1)
        # object.position[1] = np.sin(timestamp) * 0.5
        object.rotate_z(math.sin(timestamp) * 0.5)
        print(object.position)
        object.update_world_transform()
        # object.scale[:] = 3.0 + 2 * math.sin(timestamp * 2.0)
        # object.rotation_x(math.pi * 0.01)
        scene.update_world_transform()
        plt.pause(0.05)
        renderer.render(camera, scene)


if __name__ == "__main__":
    main()
