import numpy as np
from pyrr import matrix44
import math


class Object3D:
    def __init__(self, name="Object"):
        self.name = name
        self.position = np.array([0.0, 0.0, 0.0], dtype=float)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=float)  # euler in radians
        self.scale = np.array([1.0, 1.0, 1.0], dtype=float)

        self.parent = None
        self.children = []

        self.local_matrix = np.eye(4, dtype=float)
        self.world_matrix = np.eye(4, dtype=float)

    def add_child(self, child: "Object3D"):
        child.parent = self
        self.children.append(child)

    def compute_local_matrix(self):
        t = matrix44.create_from_translation(self.position)
        rx = matrix44.create_from_x_rotation(self.rotation[0])
        ry = matrix44.create_from_y_rotation(self.rotation[1])
        rz = matrix44.create_from_z_rotation(self.rotation[2])
        s = matrix44.create_from_scale(self.scale)

        # local = T * Rz * Ry * Rx * S  (common convention)
        self.local_matrix = matrix44.multiply(s, np.eye(4))
        self.local_matrix = matrix44.multiply(rx, self.local_matrix)
        self.local_matrix = matrix44.multiply(ry, self.local_matrix)
        self.local_matrix = matrix44.multiply(rz, self.local_matrix)
        self.local_matrix = matrix44.multiply(t, self.local_matrix)

    def update_world_matrix(self, parent_world=None):
        self.compute_local_matrix()

        if parent_world is None:
            self.world_matrix = self.local_matrix.copy()
        else:
            self.world_matrix = matrix44.multiply(self.local_matrix, parent_world)

        for child in self.children:
            child.update_world_matrix(self.world_matrix)


# -------------------- TESTS --------------------

def almost_equal(a, b, eps=1e-5):
    return np.allclose(a, b, atol=eps)


def run_tests():
    print("Running tests...")

    # Test 1: Basic translation
    root = Object3D("Root")
    root.position = np.array([1, 2, 3], dtype=float)
    root.update_world_matrix()
    expected_translation = matrix44.create_from_translation([1, 2, 3])
    assert almost_equal(root.world_matrix, expected_translation), "Translation test failed"

    # Test 2: Basic scaling
    obj = Object3D("Scaler")
    obj.scale = np.array([2, 2, 2], dtype=float)
    obj.update_world_matrix()
    expected_scale = matrix44.create_from_scale([2, 2, 2])
    assert almost_equal(obj.world_matrix, expected_scale), "Scale test failed"

    # Test 3: Basic rotation (90° around Z)
    obj = Object3D("Rotator")
    obj.rotation = np.array([0, 0, math.pi/2], dtype=float)
    obj.update_world_matrix()
    expected_rot = matrix44.create_from_z_rotation(math.pi/2)
    assert almost_equal(obj.world_matrix, expected_rot), "Rotation test failed"

    # Test 4: Hierarchical transformation
    parent = Object3D("Parent")
    child = Object3D("Child")
    parent.position = np.array([5, 0, 0], dtype=float)
    child.position = np.array([0, 3, 0], dtype=float)
    parent.add_child(child)
    parent.update_world_matrix()

    # Child world = Parent translation * Child translation
    expected_child_world = matrix44.create_from_translation([5, 0, 0])
    expected_child_world = matrix44.multiply(
        matrix44.create_from_translation([0, 3, 0]),
        expected_child_world
    )
    assert almost_equal(child.world_matrix, expected_child_world), "Hierarchy test failed"

    print("✅ All tests passed successfully!")


if __name__ == "__main__":
    run_tests()
