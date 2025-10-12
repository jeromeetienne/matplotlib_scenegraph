import numpy as np
from pyrr import quaternion, matrix44
from object3d import Object3D


def test_add_and_remove_child():
    parent = Object3D(name="parent")
    child = Object3D(name="child")
    parent.add_child(child)
    assert child in parent.children
    assert child.parent is parent

    parent.remove_child(child)
    assert child not in parent.children
    assert child.parent is None


def test_traverse_and_get_object_by_name():
    root = Object3D(name="root")
    a = Object3D(name="a")
    b = Object3D(name="b")
    c = Object3D(name="c")
    root.add_child(a)
    a.add_child(b)
    b.add_child(c)
    names = [node.name for node in root.traverse()]
    assert names == ["root", "a", "b", "c"]
    assert root.get_object_by_name("c") is c
    assert root.get_object_by_name("notfound") is None


def test_world_position_no_parent():
    obj = Object3D(position=np.array([1, 2, 3], dtype=float))
    np.testing.assert_allclose(obj.world_position(), [1, 2, 3])


def test_world_position_with_parent():
    parent = Object3D(position=np.array([1, 0, 0], dtype=float))
    child = Object3D(position=np.array([0, 2, 0], dtype=float))
    parent.add_child(child)
    np.testing.assert_allclose(child.world_position(), [1, 2, 0])


def test_nested_world_position_complex():
    # Build a deep hierarchy with nontrivial positions, scales, and rotations
    root = Object3D(name="root", position=np.array([10, 0, 0], dtype=float), scale=np.array([2, 2, 2], dtype=float))
    child = Object3D(name="child", position=np.array([0, 5, 0], dtype=float), scale=np.array([0.5, 1, 1], dtype=float))
    grandchild = Object3D(name="grandchild", position=np.array([0, 0, 3], dtype=float), scale=np.array([1, 0.5, 2], dtype=float))
    great_grandchild = Object3D(name="great_grandchild", position=np.array([1, 1, 1], dtype=float), scale=np.array([2, 2, 2], dtype=float))

    # Apply rotations at each level
    root.rotate_z(np.pi / 4)  # 45 deg around Z
    child.rotate_y(np.pi / 3)  # 60 deg around Y
    grandchild.rotate_x(np.pi / 6)  # 30 deg around X
    great_grandchild.rotate_z(np.pi / 2)  # 90 deg around Z

    root.add_child(child)
    child.add_child(grandchild)
    grandchild.add_child(great_grandchild)

    # Compute expected world position step by step
    # Start with great_grandchild's local position
    pos = np.array([1, 1, 1, 1], dtype=float)
    # Apply great_grandchild's scale
    pos[:3] *= great_grandchild.scale
    # Transform by grandchild
    m_grandchild = matrix44.create_from_scale(grandchild.scale)
    m_grandchild = matrix44.multiply(m_grandchild, matrix44.create_from_quaternion(grandchild.rotation))
    m_grandchild = matrix44.multiply(m_grandchild, matrix44.create_from_translation(grandchild.position))
    pos = np.dot(m_grandchild, pos)
    # Transform by child
    m_child = matrix44.create_from_scale(child.scale)
    m_child = matrix44.multiply(m_child, matrix44.create_from_quaternion(child.rotation))
    m_child = matrix44.multiply(m_child, matrix44.create_from_translation(child.position))
    pos = np.dot(m_child, pos)
    # Transform by root
    m_root = matrix44.create_from_scale(root.scale)
    m_root = matrix44.multiply(m_root, matrix44.create_from_quaternion(root.rotation))
    m_root = matrix44.multiply(m_root, matrix44.create_from_translation(root.position))
    pos = np.dot(m_root, pos)
    expected_pos = pos[:3]

    np.testing.assert_allclose(great_grandchild.world_position(), expected_pos, atol=1e-6)


def test_world_scale():
    parent = Object3D(scale=np.array([2, 2, 2], dtype=float))
    child = Object3D(scale=np.array([0.5, 1, 1], dtype=float))
    parent.add_child(child)
    np.testing.assert_allclose(child.world_scale(), [1, 2, 2])


def test_nested_world_scale_complex():
    # Create a hierarchy with varying scales
    root = Object3D(name="root", scale=np.array([2, 1, 1], dtype=float))
    child = Object3D(name="child", scale=np.array([0.5, 3, 1], dtype=float))
    grandchild = Object3D(name="grandchild", scale=np.array([4, 0.5, 2], dtype=float))
    root.add_child(child)
    child.add_child(grandchild)

    # The world scale should be the product of all ancestor scales
    expected_scale = np.array([2, 1, 1], dtype=float) * np.array([0.5, 3, 1], dtype=float) * np.array([4, 0.5, 2], dtype=float)
    np.testing.assert_allclose(grandchild.world_scale(), expected_scale)

    # Add another level with non-uniform scale
    great_grandchild = Object3D(name="great_grandchild", scale=np.array([0.25, 2, 0.5], dtype=float))
    grandchild.add_child(great_grandchild)
    expected_scale = expected_scale * np.array([0.25, 2, 0.5], dtype=float)
    np.testing.assert_allclose(great_grandchild.world_scale(), expected_scale)


def test_world_rotation():
    angle = np.pi / 2
    parent = Object3D()
    parent.rotate_z(angle)
    child = Object3D()
    parent.add_child(child)
    # The child's world rotation should match the parent's rotation
    np.testing.assert_allclose(child.world_rotation(), parent.rotation)


def test_nested_world_rotation_complex():
    # Compose rotations: root rotates 90deg Z, child 90deg Y, grandchild 90deg X
    root = Object3D(name="root")
    root.rotate_z(np.pi / 2)
    child = Object3D(name="child")
    child.rotate_y(np.pi / 2)
    grandchild = Object3D(name="grandchild")
    grandchild.rotate_x(np.pi / 2)
    root.add_child(child)
    child.add_child(grandchild)

    # Compute expected world rotation: root * child * grandchild
    expected = quaternion.cross(quaternion.cross(root.rotation, child.rotation), grandchild.rotation)
    np.testing.assert_allclose(grandchild.world_rotation(), quaternion.normalize(expected))

    # Add a fourth level with a random rotation
    angle = np.pi / 3
    great_grandchild = Object3D(name="great_grandchild")
    great_grandchild.rotate_z(angle)
    grandchild.add_child(great_grandchild)
    expected = quaternion.cross(expected, great_grandchild.rotation)
    np.testing.assert_allclose(great_grandchild.world_rotation(), quaternion.normalize(expected))


def main():
    # Build a simple scene graph
    root = Object3D(name="root", position=np.array([0, 0, 0], dtype=float), scale=np.array([1, 1, 1], dtype=float))
    child1 = Object3D(name="child1", position=np.array([1, 0, 0], dtype=float), scale=np.array([2, 2, 2], dtype=float))
    child2 = Object3D(name="child2", position=np.array([0, 1, 0], dtype=float), scale=np.array([0.5, 1, 1], dtype=float))
    root.add_child(child1)
    child1.add_child(child2)

    # Apply rotation to child1
    angle = np.pi / 2
    child1.rotate_z(angle)

    # Check world position
    pos_child2 = child2.world_position()
    # child2 is at (0,1,0) in child1's space, child1 is at (1,0,0) in root's space
    # child1 is rotated 90deg around Z, so child2's local (0,1,0) becomes (-1,0,0) in world
    # child2's world position should be child1's world position + rotated and scaled child2 position
    expected_pos = (
        np.array([1, 0, 0], dtype=float)
        + np.dot(matrix44.create_from_quaternion(child1.rotation), np.array([0, 1, 0, 1], dtype=float) * np.array([2, 2, 2, 1], dtype=float))[:3]
    )
    np.testing.assert_allclose(pos_child2, expected_pos, atol=1e-6)

    # Check world scale
    expected_scale = np.array([1, 1, 1], dtype=float) * np.array([2, 2, 2], dtype=float) * np.array([0.5, 1, 1], dtype=float)
    np.testing.assert_allclose(child2.world_scale(), expected_scale)

    # Check world rotation
    expected_rot = quaternion.normalize(quaternion.cross(child1.rotation, child2.rotation))
    np.testing.assert_allclose(child2.world_rotation(), expected_rot)

    print("All main scene graph tests passed.")


if __name__ == "__main__":
    test_add_and_remove_child()
    test_world_position_no_parent()
    test_world_position_with_parent()
    test_nested_world_position_complex()
    test_world_scale()
    test_nested_world_scale_complex()
    test_world_rotation()
    test_nested_world_rotation_complex()
    test_traverse_and_get_object_by_name()
    main()
