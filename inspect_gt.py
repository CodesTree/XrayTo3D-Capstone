import scipy.io
import numpy as np
import pathlib
import nrrd

base = pathlib.Path(r"C:\Users\Chan Zheng Shao\OneDrive\Desktop\Github Repo\TestProject\TestProject")

# --- VSD .mat: structured array with mesh data ---
for name in ["002", "z001"]:
    mat_path = base / "data" / "raw" / "baseline_models" / "VSD" / f"{name}.mat"
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    print(f"=== VSD {name}.mat ===")
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        print(f"  top-level key='{k}'  type={type(v).__name__}  shape={getattr(v, 'shape', 'N/A')}")
        # It's a structured/object array
        if hasattr(v, "dtype") and v.dtype.names:
            print(f"    struct fields: {v.dtype.names}")
            # Inspect each element
            arr = np.atleast_1d(v)
            print(f"    num entries: {len(arr)}")
            entry = arr[0]
            for field in v.dtype.names:
                fval = entry[field]
                print(f"    entry[0].{field}: type={type(fval).__name__}  shape={getattr(fval, 'shape', 'N/A')}")
                if hasattr(fval, "shape") and fval.dtype.kind in "fiu":
                    print(f"      min={fval.min():.4f}  max={fval.max():.4f}  dtype={fval.dtype}")
                elif isinstance(fval, np.ndarray) and fval.dtype.names:
                    print(f"      sub-struct fields: {fval.dtype.names}")
                    for sf in fval.dtype.names:
                        sfv = fval[sf]
                        print(f"        .{sf}: type={type(sfv).__name__}  shape={getattr(sfv, 'shape', 'N/A')}")
                        if hasattr(sfv, "dtype") and sfv.dtype.kind in "fiu":
                            print(f"          min={sfv.min():.4f}  max={sfv.max():.4f}  dtype={sfv.dtype}")
                elif isinstance(fval, str):
                    print(f"      value='{fval}'")
        elif hasattr(v, "dtype") and v.dtype.kind == "O":
            # object array
            print(f"    object array of length {len(np.atleast_1d(v))}")
            for i, item in enumerate(np.atleast_1d(v)[:3]):
                print(f"    [{i}]: type={type(item).__name__}  value={str(item)[:80]}")
    print()

# --- Fractured .seg.nrrd ---
for case in ["Baseline Case1", "Baseline Case5"]:
    seg_path = base / "data" / "raw" / "baseline_models" / "fractured" / case / "Segmentation.seg.nrrd"
    data, header = nrrd.read(str(seg_path))
    print(f"=== Fractured {case} Segmentation.seg.nrrd ===")
    print(f"  shape={data.shape}  dtype={data.dtype}")
    print(f"  min={data.min()}  max={data.max()}")
    u = np.unique(data)
    print(f"  unique values ({len(u)}): {u}")
    # Print relevant header keys
    relevant = ["dimension", "sizes", "space", "space directions", "space origin",
                "type", "encoding", "Segment0_ID", "Segment0_Name", "Segment0_Color",
                "Segment1_ID", "Segment1_Name", "Segment1_Color",
                "Segment2_ID", "Segment2_Name", "Segment2_Color",
                "Segment3_ID", "Segment3_Name", "Segment3_Color",
                "Segment4_ID", "Segment4_Name", "Segment4_Color"]
    print("  Header (relevant keys):")
    for k in relevant:
        if k in header:
            print(f"    {k}: {str(header[k])[:150]}")
    # Also print any Segment keys not in list above
    for k in header:
        if k.startswith("Segment") and k not in relevant:
            print(f"    {k}: {str(header[k])[:100]}")
    print()
