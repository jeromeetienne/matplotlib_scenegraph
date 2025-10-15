# Scene graph with matplotlib


## Supported Objects
- Points
- Lines
- Polygons
- TexturePolygons

## Able to display
- wireframe with polygons (transparent background color)
- hidden face with polygons (solid background color)
- texture with TexturePolygons
- point clouds with Points
- lines with Lines

## How to install the project

Note that this project requires Python 3.11 or higher.

Potentially create a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
```

To install the project itself, you can use pip:

```bash
pip install -e .
```

# Makefile

A Makefile is provided to help with common tasks:
- `make help`: Show available make commands
- `make test`: Run all the tests
- `make lint`: Run linting checks
- `make run_all_examples`: Run all example scripts to ensure they run without exceptions
- `make check_expected_output`: Check if the output of examples matches expected output