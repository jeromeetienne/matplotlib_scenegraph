from pyrr import Matrix44, Vector3, Quaternion
import numpy as np


class SceneNode:
    def __init__(self, name, parent=None, position=None, rotation_euler=None, scale=None):
        self.name = name
        self.parent = parent
        self.children = []

        self.position = Vector3(position) if position is not None else Vector3([0, 0, 0])
        self.rotation_euler = Vector3(rotation_euler) if rotation_euler is not None else Vector3([0, 0, 0])  # radians
        self.scale = Vector3(scale) if scale is not None else Vector3([1, 1, 1])

        self.local_transform = Matrix44.identity()
        self.global_transform = Matrix44.identity()

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    def update_local_transform(self):
        # Compose rotation from Euler angles (XYZ order)
        qx = Quaternion.from_x_rotation(self.rotation_euler.x)
        qy = Quaternion.from_y_rotation(self.rotation_euler.y)
        qz = Quaternion.from_z_rotation(self.rotation_euler.z)
        rotation = qz * qy * qx  # rotation order ZYX (can be modified as needed)

        # Compose local transform: scale * rotation * translation
        scale_matrix = Matrix44.from_scale(self.scale)
        rotation_matrix = rotation.matrix44
        translation_matrix = Matrix44.from_translation(self.position)

        # Note: Transform order matters; here translation last to apply translation in parent's space
        self.local_transform = translation_matrix * rotation_matrix * scale_matrix

    def update_global_transform(self):
        self.update_local_transform()

        if self.parent:
            self.global_transform = self.parent.global_transform * self.local_transform
        else:
            self.global_transform = self.local_transform

        for child in self.children:
            child.update_global_transform()

    def __repr__(self):
        return f"<SceneNode {self.name}, Pos: {self.position}, Rot: {self.rotation_euler}, Scale: {self.scale}>"


# Example usage

root = SceneNode("Root")
child1 = SceneNode("Child1", parent=root, position=[2, 0, 0], rotation_euler=[0, np.pi / 4, 0], scale=[1, 2, 1])
child2 = SceneNode("Child2", parent=child1, position=[0, 3, 0])

root.update_global_transform()

print(f"Root global transform:\n{root.global_transform}")
print(f"Child1 global transform:\n{child1.global_transform}")
print(f"Child2 global transform:\n{child2.global_transform}")

position = Vector3([1, 2, 3])
transformed_position = child2.global_transform * position
print("Transformed position:", transformed_position)


# Suppose geometry_points is an Nx3 numpy array (e.g., 100000 points)
geometry_points = np.random.rand(100000, 3)

# Convert Nx3 points to Nx4 by adding homogeneous coordinate 1
homogeneous_points = np.hstack([geometry_points, np.ones((geometry_points.shape[0], 1))])  # Nx4

# Get the 4x4 transformation matrix as numpy array
transform_matrix = child2.global_transform.astype(np.float64)  # Ensuring numpy array

# Multiply points by the transpose of transform matrix (to match multiplication rules)
transformed_homogeneous = homogeneous_points @ transform_matrix.T  # Nx4

# Convert back to 3D by dropping the last column
transformed_points = transformed_homogeneous[:, :3]

print(transformed_points.shape)  # Should be (100000, 3)
