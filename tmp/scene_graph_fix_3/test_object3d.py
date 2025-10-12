"""Simple standalone checks for the Object3D implementation.

Run this module directly with ``python test_object3d.py``; it does not rely on
pytest so it can be executed in constrained environments.
"""

import math
from typing import Callable

import numpy as np
from pyrr import matrix44, quaternion

from object3d import Object3D


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_add_remove_child() -> None:
    parent = Object3D()
    child = Object3D()

    parent.add_child(child)
    _check(child in parent.children, "child not registered under parent")
    _check(child.parent is parent, "child parent link incorrect")

    parent.remove_child(child)
    _check(child not in parent.children, "child still referenced after removal")
    _check(child.parent is None, "child parent link not cleared")


def test_object_name_persistence() -> None:
    node = Object3D(name="root")
    _check(node.name == "root", "Object name not stored correctly")


def test_traverse_depth_first() -> None:
    root = Object3D(name="root")
    child_a = Object3D(name="A")
    child_b = Object3D(name="B")
    grand_a1 = Object3D(name="A1")
    grand_b1 = Object3D(name="B1")

    root.add_child(child_a)
    root.add_child(child_b)
    child_a.add_child(grand_a1)
    child_b.add_child(grand_b1)

    order = [node.name for node in root.traverse()]
    _check(order == ["root", "A", "A1", "B", "B1"], "Traverse order incorrect")


def test_get_by_name() -> None:
    root = Object3D(name="root")
    target = Object3D(name="target")
    other = Object3D(name="other")

    root.add_child(target)
    target.add_child(other)

    found = root.get_by_name("target")
    _check(found is target, "get_by_name did not return expected node")
    _check(root.get_by_name("missing") is None, "get_by_name should return None when missing")


def test_local_transform_composition() -> None:
    rotation = quaternion.create_from_z_rotation(math.pi / 3, dtype=np.float64)
    node = Object3D()
    node.position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    node.rotation = rotation
    node.scale = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    node.update_world_transform()

    expected_scale = matrix44.create_from_scale(node.scale)
    expected_rot = matrix44.create_from_quaternion(node.rotation)
    expected_trans = matrix44.create_from_translation(node.position)

    expected_local = matrix44.multiply(expected_rot, expected_scale)
    expected_local = matrix44.multiply(expected_trans, expected_local)

    _check(np.allclose(node.local_transform, expected_local), "local transform composition mismatch")


def test_world_transform_with_parent() -> None:
    parent = Object3D()
    parent.position = np.array([5.0, 0.0, 0.0], dtype=np.float64)
    parent.update_world_transform()

    child = Object3D()
    child.position = np.array([0.0, 4.0, 0.0], dtype=np.float64)

    parent.add_child(child)

    expected_world = matrix44.multiply(parent.world_transform, child.local_transform)
    _check(np.allclose(child.world_transform, expected_world), "world transform mismatch for child")


def test_rotation_helpers() -> None:
    node = Object3D()

    node.rotate_x(math.pi / 2)
    expected = quaternion.create_from_x_rotation(math.pi / 2, dtype=np.float64)
    _check(np.allclose(node.rotation, quaternion.normalize(expected)), "rotate_x mismatch")

    node.rotate_y(math.pi / 4)
    expected_y = quaternion.create_from_y_rotation(math.pi / 4, dtype=np.float64)
    combo = quaternion.normalize(quaternion.cross(expected_y, quaternion.create_from_x_rotation(math.pi / 2)))
    _check(np.allclose(node.rotation, combo), "rotate_y composition mismatch")

    node.rotate_z(-math.pi / 3)
    expected_z = quaternion.create_from_z_rotation(-math.pi / 3, dtype=np.float64)
    combo = quaternion.normalize(quaternion.cross(expected_z, combo))
    _check(np.allclose(node.rotation, combo), "rotate_z composition mismatch")


def test_hierarchy_world_transform_update() -> None:
    root = Object3D()
    root.position = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    root.update_world_transform()

    child = Object3D()
    child.position = np.array([0.0, 2.0, 0.0], dtype=np.float64)

    grandchild = Object3D()
    grandchild.position = np.array([0.0, 0.0, 3.0], dtype=np.float64)

    root.add_child(child)
    child.add_child(grandchild)

    child.rotate_z(math.pi / 2)
    grandchild.scale = np.array([1.0, 2.0, 1.0], dtype=np.float64)
    grandchild.update_world_transform(child.world_transform)

    expected_child_world = matrix44.multiply(root.world_transform, child.local_transform)
    expected_grand_world = matrix44.multiply(expected_child_world, grandchild.local_transform)

    _check(
        np.allclose(child.world_transform, expected_child_world),
        "child world transform incorrect after hierarchy update",
    )
    _check(
        np.allclose(grandchild.world_transform, expected_grand_world),
        "grandchild world transform incorrect after hierarchy update",
    )


def test_deep_hierarchy_mixed_transforms() -> None:
    root = Object3D()
    root.position = np.array([2.0, -1.0, 5.0], dtype=np.float64)
    root.rotation = quaternion.create_from_z_rotation(math.radians(35.0), dtype=np.float64)
    root.scale = np.array([1.5, 1.0, 0.5], dtype=np.float64)
    root.update_world_transform()

    child = Object3D()
    child.position = np.array([1.0, 2.0, -1.0], dtype=np.float64)
    child.rotation = quaternion.create_from_y_rotation(math.radians(-20.0), dtype=np.float64)
    child.scale = np.array([0.5, 2.0, 1.0], dtype=np.float64)

    grandchild = Object3D()
    grandchild.position = np.array([-0.5, 0.0, 3.0], dtype=np.float64)
    grandchild.rotation = quaternion.create_from_x_rotation(math.radians(15.0), dtype=np.float64)
    grandchild.scale = np.array([2.0, 0.25, 1.5], dtype=np.float64)

    root.add_child(child)
    child.add_child(grandchild)

    expected_child_world = matrix44.multiply(root.world_transform, child.local_transform)
    expected_grand_world = matrix44.multiply(expected_child_world, grandchild.local_transform)

    _check(
        np.allclose(child.world_transform, expected_child_world),
        "mixed child world matrix incorrect",
    )
    _check(
        np.allclose(grandchild.world_transform, expected_grand_world),
        "mixed grandchild world matrix incorrect",
    )

    local_point = np.array([0.25, -0.75, 1.0, 1.0], dtype=np.float64)
    world_from_matrix = expected_grand_world @ local_point
    world_from_node = grandchild.world_transform @ local_point
    _check(
        np.allclose(world_from_node, world_from_matrix),
        "world matrix transform of point mismatch",
    )


def main() -> None:
    tests: list[tuple[str, Callable[[], None]]] = [
        ("add/remove child", test_add_remove_child),
        ("object name persistence", test_object_name_persistence),
        ("traverse depth first", test_traverse_depth_first),
        ("get by name", test_get_by_name),
        ("local transform composition", test_local_transform_composition),
        ("world transform with parent", test_world_transform_with_parent),
        ("rotation helpers", test_rotation_helpers),
        ("hierarchy world transform update", test_hierarchy_world_transform_update),
        ("deep hierarchy mixed transforms", test_deep_hierarchy_mixed_transforms),
    ]

    for name, func in tests:
        func()
        print(f"[PASS] {name}")

    print("All Object3D tests passed.")


if __name__ == "__main__":
    main()
