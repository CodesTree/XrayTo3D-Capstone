import numpy as np
import pathlib
import struct

base = pathlib.Path(r"C:\Users\Chan Zheng Shao\OneDrive\Desktop\Github Repo\TestProject\TestProject")

def read_stl_binary(path):
    """Read binary STL, return (n_triangles, vertices_flat, normals)."""
    with open(path, "rb") as f:
        f.read(80)  # header
        n = struct.unpack("<I", f.read(4))[0]
        triangles = []
        for _ in range(n):
            normal = struct.unpack("<3f", f.read(12))
            v1 = struct.unpack("<3f", f.read(12))
            v2 = struct.unpack("<3f", f.read(12))
            v3 = struct.unpack("<3f", f.read(12))
            f.read(2)  # attr
            triangles.append([v1, v2, v3])
        return n, np.array(triangles, dtype=np.float32)

def read_stl_ascii(path):
    """Fallback ASCII STL reader."""
    verts = []
    with open(path, "r") as f:
        for line in f:
            l = line.strip()
            if l.startswith("vertex"):
                coords = list(map(float, l.split()[1:]))
                verts.append(coords)
    verts = np.array(verts)
    n_tri = len(verts) // 3
    return n_tri, verts.reshape(n_tri, 3, 3) if len(verts) else (0, np.zeros((0,3,3)))

def inspect_stl(path):
    try:
        with open(path, "rb") as f:
            header = f.read(5)
        if header[:5] == b"solid":
            n, tris = read_stl_ascii(path)
        else:
            n, tris = read_stl_binary(path)
        if n > 0:
            all_verts = tris.reshape(-1, 3)
            bbox_min = all_verts.min(axis=0)
            bbox_max = all_verts.max(axis=0)
            dims = bbox_max - bbox_min
            return {
                "triangles": n,
                "vertices_unique_approx": n * 3,
                "bbox_min_mm": bbox_min.tolist(),
                "bbox_max_mm": bbox_max.tolist(),
                "size_mm": dims.tolist(),
            }
        return {"triangles": 0}
    except Exception as e:
        return {"error": str(e)}

print("=== Fractured Baseline STLs ===")
frac_base = base / "data" / "raw" / "baseline_models" / "fractured"
for case in ["Baseline Case1", "Baseline Case5"]:
    print(f"\n  {case}:")
    case_dir = frac_base / case
    for stl in sorted(case_dir.glob("*.stl")):
        info = inspect_stl(stl)
        if "error" in info:
            print(f"    {stl.name}: ERROR {info['error']}")
        else:
            size = [f"{s:.1f}" for s in info["size_mm"]]
            print(f"    {stl.name}")
            print(f"      triangles={info['triangles']:,}  bbox_size={size} mm [X, Y, Z]")

print("\n=== VSD .mat meshes (002.mat, knee bones only) ===")
import scipy.io
mat = scipy.io.loadmat(
    str(base / "data" / "raw" / "baseline_models" / "VSD" / "002.mat"),
    squeeze_me=True, struct_as_record=False
)
B = np.atleast_1d(mat["B"])
knee_names = {"Femur_R", "Femur_L", "Patella_R", "Patella_L", "Tibia_R", "Tibia_L", "Fibula_R", "Fibula_L"}
for b in B:
    if b.name in knee_names:
        v = b.mesh.vertices
        bbox_min = v.min(axis=0)
        bbox_max = v.max(axis=0)
        size = bbox_max - bbox_min
        print(f"  {b.name}: vertices={v.shape[0]:,}  faces={b.mesh.faces.shape[0]:,}  size={[f'{s:.1f}' for s in size]} mm")
