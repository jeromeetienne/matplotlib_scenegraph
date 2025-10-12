
# Initial Task
## Tasks
Generate a scene graph implementation only using the vscode open editors
- keep it standalone (no external dependencies except numpy and pyrr, and matplotlib for testing)
- use pyrr pip package for matrix and quaternion math
  - use the OOP API (pyrr.Matrix44, pyrr.Quaternion, pyrr.Vector3)
  - it is ok to use pyrr objects like Matrix44, Quaternion, Vector3, etc.
  - NOTE: pyrr uses row-major matrices

# 3d convention
- the scene graph will be right handed
- the transform ordering will be scale, then rotation, then translation

## object3d.py
- add class `Object3D` with local_transform and world_transform
  - position, rotation, scale (as numpy arrays, rotation handled by quaternions)
  - add_child, remove_child
  - rotate_x, rotate_y, rotate_z functions
  - get_rotation_euler() to return rotation in euler angles (in radians)
  - set_rotation_euler(euler) to set rotation from euler angles (in radians)
  - update_world_transform

## camera.py
- add class `Camera` inheriting from `Object3D`
  - fov, aspect_ratio, near, far
  - get_view_matrix
  - get_projection_matrix

## renderer.py
- add class `Renderer`
  - render(camera, scene)
- use matplotlib to render the scene from the camera's perspective
  - display all objects as simple points


## main.py 
- create a simple scene with a few objects and a camera
- use Renderer to render the scene


## IMPORTANT
- use only the file open in vscode
- dont read other files in the directory

