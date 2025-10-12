"""Basic tests for Object3D transforms."""

from __future__ import annotations

import math
import pathlib
import sys

if __package__ in (None, ""):
    current_dir = pathlib.Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.append(str(current_dir))
    from object3d import Object3D  # type: ignore[import-not-found]
else:
    from .object3d import Object3D


def test_child_translation() -> None:
    root = Object3D()
    child = Object3D()
    root.set_position((1.0, 0.0, 0.0))
    child.set_position((0.0, 2.0, 0.0))

    root.add_child(child)
    root.update_world_transform()

    world_pos = child.get_world_position()
    assert abs(world_pos[0] - 1.0) < 1e-5
    assert abs(world_pos[1] - 2.0) < 1e-5
    assert abs(world_pos[2] - 0.0) < 1e-5


def test_rotation_order() -> None:
    node = Object3D()
    node.set_scale((2.0, 1.0, 1.0))
    node.rotate_y(math.radians(90.0))
    node.set_position((0.0, 0.0, 1.0))
    node.update_world_transform()

    local = node.local_transform
    expected = [0.0, 0.0, 1.0, 1.0]
    for idx, value in enumerate(expected):
        assert abs(local[3, idx] - value) < 1e-5


if __name__ == "__main__":
    test_child_translation()
    test_rotation_order()
    print("All Object3D tests passed.")
