# matplotlib_scenegraph

Matplotlib Scenegraph (`mpl-graph`) is a pure-Python 3D scene graph and renderer that leans on Matplotlib for visualization. It provides hierarchical transforms, multiple geometry types, and a lightweight animation loop so you can prototype scientific visuals, custom plot tooling, or interactive demos without leaving the Matplotlib ecosystem.

## Highlights
- **Hierarchical scene graph:** Compose complex scenes by attaching `Object3D` children; transforms propagate in the right-handed order scale → rotation → translation.
- **Multiple primitives:** Points, lines, filled polygons, textured polygons, billboards, and meshes with basic, normal, depth, Phong, or textured materials.
- **Matplotlib-powered renderer:** Generates Matplotlib artists, optional depth sorting, and per-object pre/post rendering hooks for custom effects.
- **Animation-first workflow:** `AnimationLoop` only re-renders objects you change, supports headless test mode, and can capture videos.
- **Geometry utilities:** Procedural primitives, OBJ import helpers, and transform utilities built on `numpy` and `pyrr`.
- **Examples-ready:** A full suite of runnable scenes demonstrates cameras, controls, sprite rendering, mesh loading, and lighting experiments.

## Quickstart

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

### Run Your First Scene
```bash
python examples/basic.py
```

## Minimal Example

```python
# Create a renderer
renderer = Renderer(512, 512)

# Create the scene root
scene = Scene()

# Create a camera
camera = CameraPerspective()
scene.add_child(camera)
camera.position[2] = 5.0

# Create an animation loop
animation_loop = AnimationLoop(renderer)

# Add a rotating point cloud
geometry = Geometry(np.random.uniform(-1, 1, (1000, 3)))
material = PointsMaterial()
points = Points(geometry, material)
scene.add_child(points)

# Add a rotating point cloud
animation_loop = AnimationLoop(renderer)
@animation_loop.decorator_callback
def spin(delta_time):
		points.rotation_euler[1] += delta_time
		return [points]
```

## Project Layout

```
src/mpl_graph/
	cameras/          # Perspective & orthographic cameras built on Object3D
	core/             # Scene nodes, events, transforms, constants, textures
	geometry/         # Geometry containers, mesh helpers, procedural shapes
	materials/        # Points, lines, mesh (basic/normal/depth/phong/textured) materials
	objects/          # Scene, Points, Lines, Polygons, Mesh, Sprite primitives
	renderers/        # Renderer dispatcher + type-specific Matplotlib renderers
examples/           # End-to-end demos, controllers, animation loop utilities
tools/              # CI helpers to run examples & compare expected output
tmp/                # Experimental scene graph variants and notebooks
```

## Core Concepts
- **Object3D:** Base node with world/local matrices, traversal, and event hooks (`pre_rendering`, `post_transform`, `post_rendering`).
- **Cameras:** Perspective & orthographic variants inherit from `Object3D` and expose view/projection matrices.
- **Renderers:** `Renderer.render(scene, camera)` updates the scene graph and dispatches to type-specific renderers (`RendererPoints`, `RendererMesh`, etc.). Optional `depth_sorting` adjusts Matplotlib `zorder` per object.
- **Materials:** Lightweight wrappers that hold colors, textures, lighting parameters, and feed renderer-specific draw calls.
- **Animation loop:** `AnimationLoop` ties callbacks to frame updates, supports incremental redraw, and can emit saved videos through Matplotlib’s animation API.
- **Testing mode:** Set `MPLSC_TESTING=True` before running examples to seed RNGs, skip interactive loops, and write deterministic PNG outputs under `examples/output/`.

## Examples & Tooling

see [./examples/](./examples/) for all the examples

- `examples/basic.py`: Rotating point cloud with live animation.
- `examples/geometry_shape_example.py`: Procedural primitives (box, plane, cylinder, grid).
- `examples/textured_head.py`: Textured mesh loading via `meshio` and sprites.
- `examples/controller_camera.py`: Keyboard-driven camera controllers using the animation loop events.
- `examples/save_video.py`: Records an animation to disk.
- Use `python tools/run_all_examples.py` to execute every example and catch regressions.
- Use `python tools/check_expected_output.py` to compare freshly rendered images with the references under `examples/expected/`.

## Development Workflow
- **Lint & type check:** `make lint` (Pyright across `src/`, `examples/`, `tools/`).
- **Run all tests:** `make test` (lints + run all examples + output comparison).
- **Regenerate example outputs:** `make run_all_examples`
- **Check example outputs:** `make check_example_outputs`

## Roadmap
See `TODO.md` for active investigations: additional sanity checks, expanded geometry handling, lighting improvements, material APIs, and serialization.

## Contributing
- Fork & create a feature branch off `main`.
- Keep dependencies minimal (numpy, pyrr, matplotlib, meshio, scikit-image).
- Add or update examples when introducing new primitives or renderer behavior.
- Run `make test` before submitting a PR.
- Discuss larger changes or new dependencies in an issue first; the project aims to stay lightweight and Matplotlib-native.

## License
under MIT License, see `LICENSE.txt` for details.

# Makefile

A Makefile is provided to help with common tasks:
- `make help`: Show available make commands
- `make test`: Run all the tests
- `make lint`: Run linting checks
- `make run_all_examples`: Run all example scripts to ensure they run without exceptions
- `make check_expected_output`: Check if the output of examples matches expected output