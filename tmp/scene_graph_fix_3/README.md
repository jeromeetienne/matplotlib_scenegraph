
## Tasks
Generate a scene graph implementation only using the vscode open editors
- keep it standalone (no external dependencies except numpy and pyrr, and matplotlib for testing)
- use pyrr pip package for matrix and quaternion math
  - use only the procedural API (pyrr.matrix44, pyrr.quaternion, pyrr.vector3)
  - avoid using pyrr objects like Matrix44, Quaternion, Vector3, etc.
  - NOTE: pyrr uses row-major matrices

# 3d convention
- the scene graph will be right handed
- the transform ordering will be scale, then rotation, then translation

## object3d.py
- add class `Object3D` with local_transform and world_transform
  - position, rotation, scale (as numpy arrays, rotation handled by quaternions)
  - add_child, remove_child
  - .rotate_x, .rotate_y, .rotate_z functions
  - update_world_transform

## test_object3d.py
- write a standalone test file for Object3D
  - no framework, just assert statements, no pytest
- add tests for Object3D
  - test adding/removing children
  - test local and world transforms
  - test rotation functions
  - test update_world_transform with hierarchy

## perspective_camera.py
- add class `PerspectiveCamera` inheriting from Object3D
  - fov, aspect_ratio, near, far
  - get_projection_matrix method
  - get_view_matrix method

## debug_renderer.py
- add class `BirdRenderer`
  - .render(scene_root, camera)
  - it will use matplotlib with "3d projection"
  - use matplotlib in 3d mode to visualize the scene from a bird eyes view (not the camera view)
  - for each Object3D, draw a point at its world position and a line indicating its forward direction
  - add a text label with the object's name
- add a simple test script to create a scene graph with some objects and a camera, and render it with BirdRenderer

## IMPORTANT
- use only the file open in vscode
- dont read other files in the directory

