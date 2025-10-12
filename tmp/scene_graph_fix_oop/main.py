"""Simple demo scene for the standalone OOP scene graph."""

from __future__ import annotations

import math
import pathlib
import sys

from camera import Camera  # type: ignore[import-not-found]
from object3d import Object3D  # type: ignore[import-not-found]
from renderer import Renderer  # type: ignore[import-not-found]


def build_scene() -> Object3D:
    root = Object3D()

    offsets = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.5, 0.0),
        (0.0, -0.5, 1.0),
    ]

    for idx, offset in enumerate(offsets):
        node = Object3D()
        node.set_position(offset)
        node.rotate_y(math.radians(30.0 * idx))
        root.add_child(node)

    return root


def main() -> None:
    scene = build_scene()

    camera = Camera(fov=60.0, aspect_ratio=1.0, near=0.1, far=50.0)
    camera.set_position((0.0, 0.0, 5.0))

    renderer = Renderer()
    renderer.render(camera, scene)


if __name__ == "__main__":
    main()
