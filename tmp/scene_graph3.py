from pyrr import matrix44, Vector3
import numpy as np


class Object3D:
    def __init__(self, name="Object3D"):
        self.name = name
        self.position = Vector3([0.0, 0.0, 0.0])
        self.rotation_euler = Vector3([0.0, 0.0, 0.0])  # in radians
        self.scale = Vector3([1.0, 1.0, 1.0])
        self.parent = None
        self.children = []
        self.local_matrix = matrix44.create_identity(dtype=np.float32)
        self.world_matrix = matrix44.create_identity(dtype=np.float32)

    def set_parent(self, parent):
        self.parent = parent
        parent.children.append(self)

    def update_local_matrix(self):
        # Order: scale -> rotate -> translate
        S = matrix44.create_from_scale(self.scale, dtype=np.float32)
        Rx = matrix44.create_from_x_rotation(self.rotation_euler.x, dtype=np.float32)
        Ry = matrix44.create_from_y_rotation(self.rotation_euler.y, dtype=np.float32)
        Rz = matrix44.create_from_z_rotation(self.rotation_euler.z, dtype=np.float32)
        R = matrix44.multiply(Rz, matrix44.multiply(Ry, Rx))
        T = matrix44.create_from_translation(self.position, dtype=np.float32)

        self.local_matrix = matrix44.multiply(R, S)
        self.local_matrix = matrix44.multiply(T, self.local_matrix)

    def update_world_matrix(self, parent_world=None):
        self.update_local_matrix()

        if parent_world is None:
            self.world_matrix = self.local_matrix.copy()
        else:
            self.world_matrix = matrix44.multiply(self.local_matrix, parent_world)

        # update children
        for child in self.children:
            child.update_world_matrix(self.world_matrix)


# -------- TEST --------
def test_scene_graph():
    print("Running scene graph test...")

    root = Object3D("Root")
    root.position = Vector3([10, 0, 0])

    child = Object3D("Child")
    child.position = Vector3([5, 0, 0])
    child.set_parent(root)

    grandchild = Object3D("GrandChild")
    grandchild.position = Vector3([2, 0, 0])
    grandchild.set_parent(child)

    # Update world matrices
    root.update_world_matrix()

    # Extract world translation from matrix (last column)
    root_pos = root.world_matrix[3][:3]
    child_pos = child.world_matrix[3][:3]
    grandchild_pos = grandchild.world_matrix[3][:3]

    print("Root world position:       ", root_pos)
    print("Child world position:      ", child_pos)
    print("GrandChild world position: ", grandchild_pos)

    # Assertions
    assert np.allclose(root_pos, [10, 0, 0]), "Root world position incorrect"
    assert np.allclose(child_pos, [15, 0, 0]), "Child world position should be Root + Child local"
    assert np.allclose(grandchild_pos, [17, 0, 0]), "Grandchild world position should cascade hierarchy"

    print("✅ Scene graph hierarchy calculation is correct.")


if __name__ == "__main__":
    test_scene_graph()
