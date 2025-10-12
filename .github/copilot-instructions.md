# Copilot Instructions for matplotlib_scenegraph

## Project Overview
- **Purpose:** 3D scene graph and rendering system using matplotlib for visualization.
- **Core Objects:** Points, Lines, Polygons, TexturePolygons (see `src/mpl_graph/objects/`).
- **Rendering:** Multiple renderers for different object types (see `src/mpl_graph/renderers/`).
- **Math:** Uses numpy and pyrr for all math/transformations.
- **Scene Graph:** Hierarchical, right-handed, transform order: scale → rotation → translation.

## Key Files & Structure
- `src/mpl_graph/core/object_3d.py`: Base scene node, transform logic, child management.
- `src/mpl_graph/cameras/`: Camera classes (perspective, orthographic).
- `src/mpl_graph/objects/`: Geometric primitives and mesh objects.
- `src/mpl_graph/renderers/`: Rendering logic for each object type.
- `examples/`: Usage demos, testbeds, and sample scenes.
- `tmp/scene_graph_fix/`: Experimental/standalone scene graph implementation (minimal dependencies, see its README for conventions).

## Developer Workflows
- **Install:**
  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  pip install -e .
  ```
- **Test:**
  - Run example scenes: `ipython ./examples/textured_head.py` or similar.
  - For standalone test: `python tmp/scene_graph_fix/test_object3d.py`
- **Build:** No build step; pure Python.

## Project Conventions
- **Transforms:** Always use numpy arrays; rotations as quaternions.
- **No OOP pyrr:** Never use `pyrr.Matrix44`/`Quaternion` objects—use only procedural functions.
- **Rendering:** All rendering is via matplotlib; no OpenGL or external 3D engines.
- **Right-handed system:** All coordinates and transforms follow right-handed convention.
- **Transform order:** Always apply scale, then rotation, then translation.
- **Minimal dependencies:** Especially in `tmp/scene_graph_fix/`—only numpy, pyrr, matplotlib allowed.

## Integration & Patterns
- **Scene Graph:** Parent-child relationships managed via `add_child`/`remove_child`.
- **Cameras:** Inherit from `Object3D`, provide view/projection matrix methods.
- **Renderers:** Accept a camera and scene root, traverse and render all objects.
- **Examples:** Use `examples/` for reference on scene setup and rendering.

## Special Notes
- For experimental/standalone work, follow the stricter conventions in `tmp/scene_graph_fix/README.md`.
- Do not introduce new dependencies without discussion.
- When in doubt, check the relevant README or example for the expected pattern.

