import scipy.io
import numpy as np
import pathlib
import nrrd

base = pathlib.Path(r"C:\Users\Chan Zheng Shao\OneDrive\Desktop\Github Repo\TestProject\TestProject")

# ---- Unpack VSD .mat struct deeply ----
mat_path = base / "data" / "raw" / "baseline_models" / "VSD" / "002.mat"
mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

B = mat["B"]   # array of mat_struct
M = mat["M"]   # single mat_struct

print("=== VSD 002.mat: M (metadata) struct ===")
if hasattr(M, "_fieldnames"):
    for f in M._fieldnames:
        val = getattr(M, f)
        print(f"  M.{f}: type={type(val).__name__}  shape={getattr(val,'shape','N/A')}")
        if isinstance(val, np.ndarray) and val.dtype.kind in "fiu":
            print(f"    min={val.min():.4f}  max={val.max():.4f}  dtype={val.dtype}")
        elif isinstance(val, (str, float, int)):
            print(f"    value={val}")

print(f"\n=== VSD 002.mat: B (bone array) - {len(np.atleast_1d(B))} entries ===")
B_arr = np.atleast_1d(B)
entry = B_arr[0]
if hasattr(entry, "_fieldnames"):
    print(f"  Fields per bone: {entry._fieldnames}")
    for f in entry._fieldnames:
        val = getattr(entry, f)
        print(f"  B[0].{f}: type={type(val).__name__}  shape={getattr(val,'shape','N/A')}")
        if isinstance(val, np.ndarray) and val.dtype.kind in "fiu":
            print(f"    min={val.min():.4f}  max={val.max():.4f}  dtype={val.dtype}")
            if val.ndim == 2:
                print(f"    (likely vertices: {val.shape[0]} pts x {val.shape[1]} dims)")
        elif isinstance(val, np.ndarray) and val.dtype.kind in "O":
            print(f"    object array")
        elif isinstance(val, (str, float, int)):
            print(f"    value={val}")
        # Nested struct
        elif hasattr(val, "_fieldnames"):
            print(f"    sub-struct fields: {val._fieldnames}")
            for sf in val._fieldnames:
                sv = getattr(val, sf)
                print(f"      .{sf}: type={type(sv).__name__}  shape={getattr(sv,'shape','N/A')}")
                if isinstance(sv, np.ndarray) and sv.dtype.kind in "fiu":
                    print(f"        min={sv.min():.4f}  max={sv.max():.4f}")
                    if sv.ndim == 2:
                        print(f"        (likely: {sv.shape[0]} pts x {sv.shape[1]} dims)")

# Print name of each of the 21 bones
print("\n  All 21 bone names in B:")
for i, b in enumerate(B_arr):
    if hasattr(b, "name"):
        print(f"    [{i}] name={b.name}")
    elif hasattr(b, "_fieldnames") and "name" in b._fieldnames:
        print(f"    [{i}] name={getattr(b, 'name')}")

# ---- Unpack seg.nrrd 4D layer structure ----
print("\n=== Fractured Case1: 4D seg.nrrd layer breakdown ===")
seg_path = base / "data" / "raw" / "baseline_models" / "fractured" / "Baseline Case1" / "Segmentation.seg.nrrd"
data, header = nrrd.read(str(seg_path))
print(f"  Full shape: {data.shape}  → [layers, H, W, D]")
for layer_idx in range(data.shape[0]):
    layer = data[layer_idx]
    u = np.unique(layer)
    nonzero_voxels = np.count_nonzero(layer)
    print(f"  Layer {layer_idx}: unique labels={u}  nonzero voxels={nonzero_voxels}")

# Map label values to segment names
print("\n  Label → Segment name mapping (Case1):")
label_map = {}
i = 0
while f"Segment{i}_Name" in header:
    name = header[f"Segment{i}_Name"]
    label = header[f"Segment{i}_LabelValue"]
    layer = header[f"Segment{i}_Layer"]
    print(f"    Layer {layer}, Label {label} → '{name}'")
    i += 1

print("\n=== Fractured Case5: 4D seg.nrrd layer breakdown ===")
seg5_path = base / "data" / "raw" / "baseline_models" / "fractured" / "Baseline Case5" / "Segmentation.seg.nrrd"
data5, hdr5 = nrrd.read(str(seg5_path))
print(f"  Full shape: {data5.shape}  → [layers, H, W, D]")
for layer_idx in range(data5.shape[0]):
    layer = data5[layer_idx]
    u = np.unique(layer)
    nonzero_voxels = np.count_nonzero(layer)
    print(f"  Layer {layer_idx}: unique labels={u}  nonzero voxels={nonzero_voxels}")

print("\n  Label → Segment name mapping (Case5):")
i = 0
while f"Segment{i}_Name" in hdr5:
    name = hdr5[f"Segment{i}_Name"]
    label = hdr5[f"Segment{i}_LabelValue"]
    layer = hdr5[f"Segment{i}_Layer"]
    print(f"    Layer {layer}, Label {label} → '{name}'")
    i += 1
