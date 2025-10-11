"""
Convert OBJ file for meshio compatibility. it doesnt accept different number of vertices and texture coordinates.
"""

# pip imports
import argparse


def convert_obj_for_meshio(input_path: str, output_path: str) -> None:
    vertices = []
    tex_coords = []
    normals = []
    faces = []

    # Read original OBJ
    with open(input_path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(line.strip())
            elif line.startswith("vt "):
                tex_coords.append(line.strip())
            elif line.startswith("vn "):
                normals.append(line.strip())
            elif line.startswith("f "):
                faces.append(line.strip())

    # Map from (vertex_idx, tex_idx) to new index
    vertex_map = {}
    new_vertices = []
    new_tex_coords = []
    new_normals = []
    new_faces = []

    def parse_face_vertex(v_str):
        # formats supported:
        #  v
        #  v/vt
        #  v//vn
        #  v/vt/vn
        parts = v_str.split("/")
        v = int(parts[0])
        vt = None
        vn = None
        if len(parts) == 2:
            vt = int(parts[1]) if parts[1] != "" else None
        elif len(parts) >= 3:
            vt = int(parts[1]) if parts[1] != "" else None
            vn = int(parts[2]) if parts[2] != "" else None
        return v, vt, vn

    # Build new vertices and tex coords arrays - split as needed
    for face_line in faces:
        face_tokens = face_line.split()[1:]
        new_face_indices = []
        for vert in face_tokens:
            v_idx, vt_idx, vn_idx = parse_face_vertex(vert)
            key = (v_idx, vt_idx, vn_idx)
            if key not in vertex_map:
                vertex_map[key] = len(new_vertices) + 1
                new_vertices.append(vertices[v_idx - 1])  # v_idx is 1-based
                if vt_idx is not None:
                    new_tex_coords.append(tex_coords[vt_idx - 1])  # vt_idx 1-based
                else:
                    # If no texture coordinate, generate dummy zero
                    new_tex_coords.append("vt 0.0 0.0")
                if vn_idx is not None:
                    new_normals.append(normals[vn_idx - 1])
                else:
                    # If no normal, generate dummy zero normal
                    new_normals.append("vn 0.0 0.0 0.0")
            new_face_indices.append(vertex_map[key])
        # Create face with unified indexing: f v/vt/vn v/vt/vn v/vt/vn
        face_str = "f " + " ".join(f"{i}/{i}/{i}" for i in new_face_indices)
        new_faces.append(face_str)

    # Write new OBJ
    with open(output_path, "w") as f:
        for v in new_vertices:
            f.write(v + "\n")
        for vt in new_tex_coords:
            f.write(vt + "\n")
        for vn in new_normals:
            f.write(vn + "\n")
        for face in new_faces:
            f.write(face + "\n")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert OBJ file for meshio compatibility. it doesnt accept different number of vertices and texture coordinates."
    )
    parser.add_argument("input", help="Input OBJ file path")
    parser.add_argument("output", help="Output OBJ file path")
    args = parser.parse_args()

    convert_obj_for_meshio(args.input, args.output)
    print("Converted OBJ saved.")
