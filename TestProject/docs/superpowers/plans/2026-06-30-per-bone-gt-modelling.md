# Per-Bone Ground-Truth Modelling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert manually-segmented per-bone STL ground truth (femur, tibia, patella, fibula) into grid-aligned `.nii.gz` volumes, then verify the shared encoder + U-Net/V-Net decoder can learn and predict each bone independently as a 4-channel multi-label task, with a 3D predicted-vs-GT viewer.

**Architecture:** Voxelize each STL onto its case's predrr NIfTI affine (`world→voxel`) → per-bone `.nii.gz` cache → alignment-validate → multi-label (4-ch sigmoid) head on the existing decoder → overfit sanity-check → 3D mesh comparison against the original GT STLs.

**Tech Stack:** Python 3 (`.venv`), nibabel, trimesh (new dep), scipy/skimage, numpy, PyTorch, matplotlib. Notebook-first per CLAUDE.md.

> **Notebook-first adaptation:** This project mandates `.ipynb` development (CLAUDE.md) and has no pytest suite for the pipeline. So "tests" here are **in-notebook validation cells** that `assert` on real data and print/plot evidence. Each task still follows write-check-first → implement → verify → commit. Use `NotebookEdit` to add cells. Run cells with the project interpreter `./.venv/Scripts/python.exe` (memory: default `python` has a broken numpy/matplotlib ABI). The existing working binary pipeline in `decoder_pipeline.ipynb` is **not modified**; Component 3/4 live in a new self-contained notebook.

**Canonical conventions (use everywhere):**
- Bone channel order: `BONES = ["femur", "tibia", "patella", "fibula"]`.
- predrr paths: `data/interim/predrr/fractured/<KEY>.nii.gz` where KEY ∈ {`Case1_PartLeft`,`Case2_PartLeft`,`Case3_PartLeft`,`Case5_PartRight`,`Case6_PartRight`,`Case7_PartRight`,`Case9_PartRight`,`Case11_PartRight`,`Case12_PartRight`,`Case13_PartRight`,`Case16_PartRight`}. (Case14/Case15 predrr exist but have no STL — skip.)
- STL root: `data/external/ground_truth/fracture_ground_truth/Case<N>[ suffix]/`.
- Per-bone GT output: `data/interim/gt_per_bone_256/fractured/<KEY>/<KEY>_<bone>.nii.gz` (predrr affine+header).

---

## Task 0: Add trimesh dependency

**Files:**
- Modify: `requirements.txt` (append `trimesh`)

- [ ] **Step 1: Verify trimesh is absent, then install into the project venv**

Run:
```bash
cd "C:/Users/Chan Zheng Shao/OneDrive/Desktop/Github Repo/TestProject/TestProject"
./.venv/Scripts/python.exe -m pip install trimesh
```
Expected: installs `trimesh` (+ deps). No `embree`/`rtree` needed — we use the voxel-fill path.

- [ ] **Step 2: Verify import + voxelization works**

Run:
```bash
./.venv/Scripts/python.exe -c "import trimesh,numpy as np; m=trimesh.creation.box((10,10,10)); vg=m.voxelized(1.0).fill(); print('trimesh',trimesh.__version__,'filled_cells',len(vg.points))"
```
Expected: prints a version and `filled_cells` ≈ 1000 (non-zero).

- [ ] **Step 3: Record the dependency**

Append a line `trimesh` to `requirements.txt` (keep alphabetic/existing style). Do NOT touch `requirements_hpc.txt` unless asked.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: add trimesh for STL voxelization"
```

---

## Task 1: STL filename parser + case→predrr-key mapping

**Files:**
- Create: `notebooks/modeling/gt_per_bone.ipynb` (cells: header markdown, imports/config, parser, validation)

- [ ] **Step 1: Create the notebook with a header + config cell**

Markdown cell:
```
# Per-Bone Ground-Truth Build & Alignment

Voxelize manual Slicer STLs (femur/tibia/patella/fibula) onto each case's predrr grid,
write per-bone `.nii.gz` for Slicer inspection, and validate alignment before modelling.
```
Code cell (config):
```python
from pathlib import Path
import re, numpy as np, nibabel as nib, trimesh
import matplotlib.pyplot as plt

ROOT = Path.cwd()
while not (ROOT / "data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

BONES = ["femur", "tibia", "patella", "fibula"]
STL_ROOT  = ROOT / "data/external/ground_truth/fracture_ground_truth"
PREDRR    = ROOT / "data/interim/predrr/fractured"
OUT_ROOT  = ROOT / "data/interim/gt_per_bone_256/fractured"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# folder-stem (Case<N>) -> predrr KEY (side comes from predrr, not the STL token)
CASE_TO_KEY = {p.name.split("_Part")[0]: p.stem.replace(".nii", "")
               for p in PREDRR.glob("*.nii.gz")}
print("predrr keys:", sorted(CASE_TO_KEY.values()))
```

- [ ] **Step 2: Write the parser as a validation-first cell (the "failing test")**

Add a code cell that asserts the parser maps **every** STL in **every** case folder to exactly one bone, before the parser exists — run it to see it fail with `NameError: parse_bone`:
```python
def _case_stem(folder_name: str) -> str:
    # "Case2 (does not seem fractured)" -> "Case2" ; "Case11" -> "Case11"
    return re.match(r"(Case\d+)", folder_name).group(1)

cases = {}
for folder in sorted(STL_ROOT.iterdir()):
    if not folder.is_dir():
        continue
    stem = _case_stem(folder.name)
    if stem not in CASE_TO_KEY:        # e.g. no predrr -> skip
        print("SKIP (no predrr):", folder.name); continue
    bone_map = {}
    for stl in folder.glob("*.stl"):
        bone = parse_bone(stl.name)    # <-- defined next step
        assert bone in BONES, f"bad bone {bone} for {stl.name}"
        assert bone not in bone_map, f"duplicate {bone} in {folder.name}: {stl.name} vs {bone_map[bone].name}"
        bone_map[bone] = stl
    missing = set(BONES) - set(bone_map)
    print(f"{folder.name:45s} -> {CASE_TO_KEY[stem]:18s} bones={sorted(bone_map)} missing={sorted(missing)}")
    cases[CASE_TO_KEY[stem]] = bone_map
assert cases, "no cases mapped"
```
Run the notebook up to here. Expected: `NameError: name 'parse_bone' is not defined`.

- [ ] **Step 3: Implement `parse_bone` in a cell above the validation cell**

```python
def parse_bone(fname: str) -> str:
    """Map a messy STL filename to one of BONES by substring, ignoring side/segmentation noise."""
    s = fname.lower()
    # order matters only for safety; bone tokens are mutually exclusive in this dataset
    for bone in BONES:
        if bone in s:                  # 'femur','tibia','patella','fibula'
            return bone
    raise ValueError(f"no bone token in {fname!r}")
```

- [ ] **Step 4: Re-run the validation cell — expect PASS**

Expected: one printed line per case, every case shows `bones=['femur','fibula','patella','tibia']`, `missing=[]`, and no `AssertionError`. If any case prints `missing=[...]`, STOP and report which bone STL is absent before continuing.

- [ ] **Step 5: Commit**

```bash
git add notebooks/modeling/gt_per_bone.ipynb
git commit -m "feat: parse per-bone STL filenames and map cases to predrr keys"
```

---

## Task 2: Port predrr transform + CT-replay validation gate

> **REVISED 2026-06-30.** The predrr `.nii.gz` affine has zeroed translation (world coords lost), so GT cannot be voxelized onto the predrr affine. Instead we replay the predrr geometric transform. This task ports that transform and proves the port is faithful by reproducing the stored predrr from raw CT.

**Files:**
- Modify: `notebooks/modeling/gt_per_bone.ipynb` (remove the broken direct-affine `voxelize_bone` + overlay cells from commit `5f22511`; add ported transform + `process_ct` + CT-match validation)

- [ ] **Step 1: Remove the two broken cells**

Delete the `voxelize_bone(stl_path, predrr_img)` cell and the single-case overlay cell added in commit `5f22511` (they assumed a world-registered predrr affine, which does not exist). Keep cells 0–3 (header, config, `parse_bone`, validation building `cases`).

- [ ] **Step 2: Add SimpleITK import + ported constants/functions (verbatim copy)**

Add a cell that `import SimpleITK as sitk` and `from scipy import ndimage`, then copy these **verbatim** from `notebooks/pre-processing/predrr_preprocessing.ipynb` (do NOT edit that file — copy out of it):
- From cell 1: constants `TARGET_SIZE=256, RESAMPLE_SPACING=0.5, ORIENTATION="RAS", FOV_MM=200.0, FOV_VOXELS=int(round(FOV_MM/RESAMPLE_SPACING)), HU_MIN=-450, HU_MAX=1050, ROI_INTENSITY_THRESHOLD=0.1, ROI_CLOSING_RADIUS=3, ROI_PAD_MARGIN=5`.
- From cell 3: `BONE_THR=0.45, MIN_AREA=30, END_FRAC=0.25, SI_FLIP_OVERRIDE` (the full dict), and functions `_blobs_in_axial`, `end_blob_counts`, `heuristic_flipped`, `correct_orientation`.
- From cell 6: functions `resample_volume`, `orient_volume`, `apply_bone_window`, `body_envelope_mask`, `center_to_fixed_fov`, `resize_volume`.

- [ ] **Step 3: Add the two modified/new helpers**

```python
def roi_bone_crop_idx(arr_windowed, threshold=ROI_INTENSITY_THRESHOLD,
                      closing_radius=ROI_CLOSING_RADIUS, pad=ROI_PAD_MARGIN):
    """Same as predrr roi_bone_crop but ALSO returns the crop indices so labels can reuse them."""
    mask = (arr_windowed > threshold).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 1)
    struct = ndimage.iterate_structure(struct, closing_radius)
    mask = ndimage.binary_closing(mask, structure=struct).astype(np.uint8)
    labeled, num = ndimage.label(mask)
    if num == 0:
        z1, y1, x1 = arr_windowed.shape
        return arr_windowed, (0, z1, 0, y1, 0, x1)
    sizes = ndimage.sum(mask, labeled, range(1, num + 1))
    mask = (labeled == (np.argmax(sizes) + 1)).astype(np.uint8)
    coords = np.argwhere(mask)
    z_min, y_min, x_min = coords.min(0)
    z_max, y_max, x_max = coords.max(0) + 1
    z_min = max(0, z_min - pad); y_min = max(0, y_min - pad); x_min = max(0, x_min - pad)
    z_max = min(arr_windowed.shape[0], z_max + pad)
    y_max = min(arr_windowed.shape[1], y_max + pad)
    x_max = min(arr_windowed.shape[2], x_max + pad)
    return arr_windowed[z_min:z_max, y_min:y_max, x_min:x_max], (z_min, z_max, y_min, y_max, x_min, x_max)

def resize_label(arr, target_size=TARGET_SIZE):
    """Nearest-neighbour resize for binary labels (mirrors resize_volume's geometry)."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    target = [target_size] * 3
    osz, osp = img.GetSize(), img.GetSpacing()
    nsp = [osp[i] * osz[i] / target[i] for i in range(3)]
    r = sitk.ResampleImageFilter()
    r.SetSize(target); r.SetOutputSpacing(nsp)
    r.SetOutputOrigin(img.GetOrigin()); r.SetOutputDirection(img.GetDirection())
    r.SetInterpolator(sitk.sitkNearestNeighbor); r.SetDefaultPixelValue(0.0)
    r.SetTransform(sitk.Transform())
    return sitk.GetArrayFromImage(r.Execute(img))

# Source each case from the SAME volume predrr was built from (per preprocessing metadata).
# Most fractured cases come from raw DICOM, but the cast-removed cases (Case3, Case16) were
# preprocessed from data/interim/fractured_cast_cleaned/*.nii.gz. Sourcing raw DICOM for those
# two gives a different bone window -> different ROI crop -> CT-replay Dice ~0.3. Using the
# recorded source makes the replayed crop/FOV match the stored predrr exactly.
import pandas as pd
_PRE_META = pd.read_csv(ROOT / "data/interim/predrr/preprocessing_metadata.csv").set_index("volume_id")

def _reroot(p):
    parts = Path(p).parts
    return ROOT.joinpath(*parts[parts.index("data"):]) if "data" in parts else Path(p)

def load_source_ct(key):
    """Load the exact source volume predrr used: cast-cleaned NIfTI or raw DICOM series."""
    row = _PRE_META.loc[key]
    src = _reroot(row["source_path"])
    if row["source_format"] == "nifti":
        return sitk.ReadImage(str(src))
    rd = sitk.ImageSeriesReader()
    rd.SetFileNames(rd.GetGDCMSeriesFileNames(str(src)))
    return rd.Execute()

def process_ct(key):
    """Replay predrr's geometric transform on the source CT. Returns the resampled+oriented sitk
    image (for label voxelization), the crop box, the S-I flip decision, and the final CT array."""
    img = load_source_ct(key)
    img = resample_volume(img)             # linear, 0.5mm iso
    img = orient_volume(img)               # RAS
    img_ro = img                           # geometry labels will be voxelized onto
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    arr = apply_bone_window(arr)
    arr = body_envelope_mask(arr)
    cropped, crop = roi_bone_crop_idx(arr)
    boxed, _ = center_to_fixed_fov(cropped, FOV_VOXELS)
    resized = resize_volume(boxed, TARGET_SIZE)
    final, flags = correct_orientation(key, resized)
    return dict(img_ro=img_ro, crop=crop, si_flip=flags["si_flipped"], ct_final=final)
```

- [ ] **Step 4: CT-replay validation gate (the keystone "test")**

```python
def dice(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

key = sorted(cases)[0]
ct = process_ct(key)
pred = np.asarray(nib.load(PREDRR / f"{key}.nii.gz").dataobj).astype(np.float32)
print("ct_final", ct["ct_final"].shape, "predrr", pred.shape, "si_flip", ct["si_flip"])
d = dice(ct["ct_final"] > 0.4, pred > 0.4)
print(f"CT-replay bone-mask Dice vs stored predrr: {d:.4f}")
assert d >= 0.99, f"CT replay does not reproduce predrr (Dice {d:.4f}) — transform port is not faithful"
print("KEYSTONE PASS: ported transform reproduces predrr -> labels will align by construction")
```
Expected: shapes both `(256,256,256)`, Dice ≥ 0.99. If the Dice is low, first check array layout: predrr's `dataobj` and `ct_final` must be in the same (z,y,x) order. If `process_single_volume` saved a transposed array, transpose `ct_final` to match before comparing (and apply the same layout when writing GT in Task 3). Do NOT proceed until this passes.

- [ ] **Step 5: Commit**

```bash
git add notebooks/modeling/gt_per_bone.ipynb
git commit -m "feat: port predrr transform + CT-replay validation gate"
```

---

## Task 3: Voxelize STL on raw grid + thread labels through the transform -> per-bone nii.gz

**Files:**
- Modify: `notebooks/modeling/gt_per_bone.ipynb`
- Create (output): `data/interim/gt_per_bone_256/fractured/<KEY>/<KEY>_<bone>.nii.gz`, `.../gt_per_bone_metadata.csv`

- [ ] **Step 1: Add the voxelizer + label-pipeline functions**

```python
def voxelize_on(stl_path, ref_img):
    """Solid-rasterize an STL (world LPS mm) onto ref_img's grid -> uint8 (z,y,x) array."""
    mesh = trimesh.load(stl_path, process=False)
    v = np.asarray(mesh.vertices)
    idx = np.array([ref_img.TransformPhysicalPointToContinuousIndex(tuple(map(float, p))) for p in v])
    mesh.vertices = idx[:, ::-1]                      # (i,j,k)->(k,j,i)=(z,y,x) array space
    pts = np.round(np.asarray(mesh.voxelized(pitch=1.0).fill().points)).astype(int)
    shape = sitk.GetArrayFromImage(ref_img).shape     # (z,y,x)
    m = np.zeros(shape, np.uint8)
    ok = (pts >= 0).all(1) & (pts[:, 0] < shape[0]) & (pts[:, 1] < shape[1]) & (pts[:, 2] < shape[2])
    pts = pts[ok]; m[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    return m

def label_through_pipeline(label_ro, crop, si_flip):
    """Apply predrr's post-orient numpy ops (crop/fov/resize/flip) to a label voxelized on img_ro."""
    z0, z1, y0, y1, x0, x1 = crop
    cropped = label_ro[z0:z1, y0:y1, x0:x1]
    boxed, _ = center_to_fixed_fov(cropped, FOV_VOXELS, fill_value=0.0)
    resized = resize_label(boxed, TARGET_SIZE)
    if si_flip:
        resized = np.ascontiguousarray(np.flip(resized, axis=0))
    return (resized > 0.5).astype(np.uint8)

def build_bone_gt(key, bone_map):
    """Return {bone: 256^3 uint8 mask in sitk (z,y,x) order on the predrr grid} for one case."""
    ct = process_ct(key)
    out = {}
    for bone, stl in bone_map.items():
        lab_ro = voxelize_on(stl, ct["img_ro"])
        out[bone] = label_through_pipeline(lab_ro, ct["crop"], ct["si_flip"])
    return ct, out

# --- Layout convention (from the Task 2 finding) ---
# process_ct / build_bone_gt produce arrays in SimpleITK (z,y,x) order, same as predrr's
# arr_final at save time. predrr was written via `sitk.GetImageFromArray(arr_zyx)` + WriteImage,
# so on disk nib.load(predrr).dataobj == arr_zyx.transpose(2,1,0). To make GT overlay predrr
# (in Slicer AND when later loaded by nibabel in the modelling notebook), save GT the SAME way —
# via sitk, copying predrr's geometry — and load everything here in (z,y,x) via sitk so all
# internal comparisons/overlays share one order (matching ct_final).
def save_grid_nii(arr_zyx, key, path):
    im = sitk.GetImageFromArray(arr_zyx.astype(np.uint8))
    im.CopyInformation(sitk.ReadImage(str(PREDRR / f"{key}.nii.gz")))
    sitk.WriteImage(im, str(path))

def load_grid_zyx(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
```

- [ ] **Step 2: Single-case overlay spot-check (visual gate, saved to PNG)**

```python
key = sorted(cases)[0]
ct, gt = build_bone_gt(key, cases[key])
ctf = ct["ct_final"]
for bone in BONES:
    print(key, bone, "voxels:", int(gt[bone].sum()))
    assert gt[bone].sum() > 300, f"{bone} mask implausibly small"

fig, ax = plt.subplots(len(BONES), 3, figsize=(11, 3.2 * len(BONES)))
for r, bone in enumerate(BONES):
    mask = gt[bone]
    for a, axis in enumerate([0, 1, 2]):
        s = mask.sum(axis=tuple(i for i in range(3) if i != axis)).argmax()
        ax[r, a].imshow(np.take(ctf, s, axis=axis).T, cmap="gray", origin="lower")
        ms = np.take(mask, s, axis=axis)
        ax[r, a].imshow(np.ma.masked_where(ms.T == 0, ms.T), cmap="autumn", alpha=0.5, origin="lower")
        ax[r, a].set_title(f"{key} {bone} ax{axis}", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_ROOT / f"_spotcheck_{key}.png", dpi=110); plt.show()
```
Each bone mask must sit on the bright bone of the replayed CT. (Controller will inspect this PNG.)

- [ ] **Step 3: Batch-build all cases -> per-bone nii.gz + metadata**

```python
import pandas as pd
rows = []
for key, bone_map in sorted(cases.items()):
    ct, gt = build_bone_gt(key, bone_map)
    # both in sitk (z,y,x) order -> compare directly
    d = dice(ct["ct_final"] > 0.4, load_grid_zyx(PREDRR / f"{key}.nii.gz") > 0.4)
    case_dir = OUT_ROOT / key; case_dir.mkdir(parents=True, exist_ok=True)
    for bone in BONES:
        mask = gt[bone]
        save_grid_nii(mask, key, case_dir / f"{key}_{bone}.nii.gz")   # sitk save -> overlays predrr
        rows.append(dict(key=key, bone=bone, voxels=int(mask.sum()),
                         ct_replay_dice=round(float(d), 4), source_stl=bone_map[bone].name))
    print(f"{key}: ct-replay Dice {d:.4f}  bones " + " ".join(f"{b}={int(gt[b].sum())}" for b in BONES))
meta = pd.DataFrame(rows); meta.to_csv(OUT_ROOT / "gt_per_bone_metadata.csv", index=False)
meta.groupby("key").agg(min_voxels=("voxels", "min"), ct_dice=("ct_replay_dice", "first"))
```

- [ ] **Step 4: Validate the batch (the "test" cell)**

```python
assert (meta.groupby("key").size() == 4).all(), "every case must have 4 bone files"
assert (meta.voxels > 300).all(), f"empty/tiny masks:\n{meta[meta.voxels <= 300]}"
assert (meta.ct_replay_dice >= 0.99).all(), f"CT replay failed for:\n{meta[meta.ct_replay_dice < 0.99]}"
nfiles = sum(1 for _ in OUT_ROOT.rglob('*.nii.gz'))
assert nfiles == 4 * len(cases), f"expected {4*len(cases)} files, found {nfiles}"
print("OK:", len(cases), "cases x4 bones; all CT-replays >=0.99; ", nfiles, "nii.gz written")
```

- [ ] **Step 5: Human Slicer gate**

Open one case in 3D Slicer: load `data/interim/predrr/fractured/<KEY>.nii.gz` and its 4 `gt_per_bone_256/fractured/<KEY>/<KEY>_<bone>.nii.gz`. Confirm each bone overlays the right anatomy. (Controller pauses here for the user.)

- [ ] **Step 6: Commit**

```bash
git add notebooks/modeling/gt_per_bone.ipynb data/interim/gt_per_bone_256/fractured/gt_per_bone_metadata.csv
git commit -m "feat: build aligned per-bone GT nii.gz via pipeline-replay"
```
(Decide with the user whether to commit the `.nii.gz` volumes or gitignore them; default: gitignore the volumes, commit only the metadata CSV.)

---

## Task 4: Alignment validation gate (GT-union vs predrr + MIP overlays)

**Files:**
- Modify: `notebooks/modeling/gt_per_bone.ipynb`
- Create (output): `data/interim/gt_per_bone_256/fractured/gt_per_bone_alignment.csv`, `.../alignment_grid.png`

- [ ] **Step 1: Union-vs-predrr Dice across all cases**

```python
ar = []
for key in sorted(cases):
    union = np.zeros((TARGET_SIZE,) * 3, bool)
    for bone in BONES:
        union |= load_grid_zyx(OUT_ROOT / key / f"{key}_{bone}.nii.gz") > 0
    pred_bone = load_grid_zyx(PREDRR / f"{key}.nii.gz") > 0.4
    ar.append(dict(key=key, union_voxels=int(union.sum()),
                   dice_union_vs_predrr=round(float(dice(union, pred_bone)), 4)))
align = pd.DataFrame(ar); align.to_csv(OUT_ROOT / "gt_per_bone_alignment.csv", index=False)
align
```
Note: union-vs-predrr Dice will be high but not 1.0 — predrr's bone mask includes marrow/threshold effects the STL surface segmentation may exclude, and the STL is a clean manual segmentation. Expect ~0.7–0.9; the real proof of alignment is the CT-replay Dice (Task 3) and the overlays. Flag any case markedly below the cohort median.

- [ ] **Step 2: AP/LAT MIP overlay grid (GT union over replayed-CT MIP), saved to PNG**

```python
n = len(cases); fig, ax = plt.subplots(n, 2, figsize=(8, 3.0 * n))
for r, key in enumerate(sorted(cases)):
    ctf = load_grid_zyx(PREDRR / f"{key}.nii.gz")   # predrr == ct_final (Dice 1.0); avoids DICOM reload
    union = np.zeros((TARGET_SIZE,) * 3, bool)
    for bone in BONES:
        union |= load_grid_zyx(OUT_ROOT / key / f"{key}_{bone}.nii.gz") > 0
    for c, axis, name in [(0, 1, "AP"), (1, 2, "LAT")]:
        ax[r, c].imshow(ctf.max(axis=axis).T, cmap="gray", origin="lower")
        u = union.max(axis=axis)
        ax[r, c].imshow(np.ma.masked_where(u.T == 0, u.T), cmap="autumn", alpha=0.45, origin="lower")
        ax[r, c].set_title(f"{key} {name}", fontsize=8); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT_ROOT / "alignment_grid.png", dpi=110); plt.show()
```

- [ ] **Step 3: Decide pass/fail**

```python
med = align.dice_union_vs_predrr.median()
suspect = align[align.dice_union_vs_predrr < med - 0.15]
print("cohort median union-Dice:", round(med, 3), "| suspect cases:", list(suspect.key) or "none")
```
The controller inspects `alignment_grid.png`: every GT union must overlay the bone silhouette in both AP and LAT. Because alignment is guaranteed by the CT-replay construction, a misaligned case here almost certainly means that case's STL was segmented in a *different* frame (e.g., the cast-removed Case3/Case16) — investigate those specifically if flagged. **Do not advance to Task 5 until the overlays pass.**

- [ ] **Step 4: Commit**

```bash
git add notebooks/modeling/gt_per_bone.ipynb data/interim/gt_per_bone_256/fractured/gt_per_bone_alignment.csv
git commit -m "feat: GT alignment validation (union-vs-predrr Dice + MIP overlays)"
```

## Task 5: Multi-label model + per-bone GT loader (new notebook)

**Files:**
- Create: `notebooks/modeling/gt_per_bone_modelling.ipynb`

This notebook is **self-contained**: copy the proven encoder/fusion/decoder/loss code from `decoder_pipeline.ipynb` and apply the small generalizations below. Do not edit `decoder_pipeline.ipynb`.

- [ ] **Step 1: Copy setup + model cells from `decoder_pipeline.ipynb`**

Copy these cells verbatim into the new notebook: imports/config (cell 2–4), backbone + `BiPlanarFeatureFusion` (cell 6), the model blocks `DoubleConv`/`VNetResBlock`/`SuperResHead`/`Decoder3D`/`ReconModel` (cell 11), `build_model` (cell 12). Set `MODEL="unet"` for the sanity-check.

- [ ] **Step 2: Add the multi-label config + generalize the output heads**

In the copied config cell add:
```python
N_CLASSES = 4
BONES = ["femur", "tibia", "patella", "fibula"]
```
Edit `SuperResHead.__init__`: change `self.out = nn.Conv3d(8, 1, 1)` → `self.out = nn.Conv3d(8, N_CLASSES, 1)`.
Edit `Decoder3D.__init__` aux heads: `self.aux3 = nn.Conv3d(c2, 1, 1)` → `nn.Conv3d(c2, N_CLASSES, 1)` (same for `aux2`, `aux1`).
Everything else (encoder/fusion) is unchanged — output tensors become `(B, N_CLASSES, T, T, T)`.

- [ ] **Step 3: Derive the case list from the GT cache + rebuild the STL map**

This notebook is independent of `gt_per_bone.ipynb`, so rebuild the keys and STL map locally (reuse the same parser logic). `KEYS` = cases that actually have a 4-bone GT cache:
```python
import re
STL_ROOT = ROOT / "data/external/ground_truth/fracture_ground_truth"
GT_PB    = ROOT / "data/interim/gt_per_bone_256/fractured"
PREDRR   = ROOT / "data/interim/predrr/fractured"

def parse_bone(fname):
    s = fname.lower()
    for b in BONES:
        if b in s: return b
    raise ValueError(fname)

KEYS = sorted(p.name for p in GT_PB.iterdir() if p.is_dir())   # e.g. ['Case1_PartLeft', ...]
CASE_TO_KEY = {p.name.split('_Part')[0]: p.name for p in GT_PB.iterdir() if p.is_dir()}
case_stl = {}                                                  # key -> {bone: stl_path}
for folder in STL_ROOT.iterdir():
    if not folder.is_dir(): continue
    stem = re.match(r"(Case\d+)", folder.name).group(1)
    if stem not in CASE_TO_KEY: continue
    case_stl[CASE_TO_KEY[stem]] = {parse_bone(s.name): s for s in folder.glob("*.stl")}
print(len(KEYS), "keys:", KEYS)
```

- [ ] **Step 4: Write the per-bone GT loader + build the paired index/dataset**

Per-bone loader — **mirrors `load_gt_occupancy` exactly** (threshold + nearest-downsample to
`TARGET_RES`), but loads the 4 bone files and stacks them. GT was saved via sitk CopyInformation
from predrr, so `nib.load(GT).get_fdata()` is in the same convention as `load_gt_occupancy` uses for
predrr — no transpose needed.
```python
GT_PB = ROOT / "data/interim/gt_per_bone_256/fractured"

def key_from(dataset, case, side):
    Side = "Right" if str(side).lower().startswith("r") else "Left"
    return f"{case}_Part{Side}"                       # fractured predrr/GT key

def load_gt_per_bone(dataset, case, side):
    key = key_from(dataset, case, side)
    chans = []
    for b in BONES:
        vol = nib.load(str(GT_PB / key / f"{key}_{b}.nii.gz")).get_fdata().astype(np.float32)
        occ = (vol > 0.5).astype(np.float32)
        t = F.interpolate(torch.from_numpy(occ)[None, None], size=(TARGET_RES,) * 3, mode="nearest")
        chans.append(t[0, 0])
    return torch.stack(chans)                          # (4, T, T, T)
```
Copy `build_paired_index()` and `PairedDRRVolumeDataset` from `decoder_pipeline.ipynb`, then adapt:
(a) restrict the index to fractured cases that have per-bone GT —
`paired_index = paired_index[paired_index.apply(lambda r: r.dataset=="fractured" and key_from(r.dataset,r.case,r.side) in KEYS, axis=1)].reset_index(drop=True)`;
(b) in `__getitem__`, replace the GT line with `gt = load_gt_per_bone(r.dataset, r.case, r.side)` so
`batch["gt"]` is `(B,4,T,T,T)`. Reuse `load_drr`, `paired_tf` unchanged — inputs are still AP+LAT.
(Note: only fractured has per-bone GT; the sanity-check is fractured-only.)

- [ ] **Step 5: Validate shapes (the "test" cell)**

```python
m = build_model().to(DEVICE).eval()
ap = torch.randn(1, 3, 256, 256).to(DEVICE); lat = torch.randn(1, 3, 256, 256).to(DEVICE)
with torch.no_grad():
    out = m(ap, lat)
out0 = out[0] if isinstance(out, (tuple, list)) else out
assert out0.shape[1] == 4, f"expected 4 output channels, got {out0.shape}"
assert out0.shape[2:] == (TARGET_RES,) * 3, f"expected {TARGET_RES}^3, got {out0.shape}"
r0 = paired_index.iloc[0]
gt = load_gt_per_bone(r0.dataset, r0.case, r0.side)
assert gt.shape == (4,) + (TARGET_RES,) * 3, gt.shape
print("model out:", tuple(out0.shape), "gt:", tuple(gt.shape), "| TARGET_RES", TARGET_RES)
```
Expected: `out` is `(1, 4, TARGET_RES, TARGET_RES, TARGET_RES)`, gt is `(4, TARGET_RES, TARGET_RES, TARGET_RES)`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/modeling/gt_per_bone_modelling.ipynb
git commit -m "feat: multi-label (4-bone) head + per-bone GT loader"
```

---

## Task 6: Per-channel loss/metrics + learnability sanity-check

**Files:**
- Modify: `notebooks/modeling/gt_per_bone_modelling.ipynb`

- [ ] **Step 1: Generalize loss + metrics to per-channel**

```python
class DiceBCEMC(nn.Module):
    """Multi-channel BCE + soft-Dice, averaged over the 4 bone channels."""
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
    def _dice(self, logits, target):
        p = torch.sigmoid(logits.float()); t = target.float()
        p = p.reshape(p.size(0), p.size(1), -1); t = t.reshape(t.size(0), t.size(1), -1)
        inter = (p * t).sum(-1); psum = p.sum(-1); tsum = t.sum(-1)
        d = (2 * inter + self.smooth) / (psum + tsum + self.smooth)
        return 1 - d.mean()
    def forward(self, logits, target):
        return 0.5 * F.binary_cross_entropy_with_logits(logits, target.float()) + 0.5 * self._dice(logits, target)

LOSS = DiceBCEMC()
def total_loss_mc(output, target, aux_weight=0.3):
    out = output[0] if isinstance(output, (tuple, list)) else output
    loss = LOSS(out, target)
    if isinstance(output, (tuple, list)):
        for a in output[1:]:
            loss = loss + aux_weight * LOSS(a, target)
    return loss

def per_bone_dice(logits, target, thr=0.5):
    p = (torch.sigmoid(logits.float()) > thr).float()
    p = p.reshape(p.size(0), p.size(1), -1); t = target.float().reshape(target.size(0), target.size(1), -1)
    inter = (p * t).sum(-1); s = p.sum(-1) + t.sum(-1)
    d = (2 * inter + 1e-6) / (s + 1e-6)        # (B,4)
    return d.mean(0).cpu().numpy()             # per-bone mean over batch
```

- [ ] **Step 2: Build the overfit sanity-check run**

```python
SANITY_KEYS = KEYS[:3]                          # ~3 fractured cases; train == val == these
train_ds = PairedDRRVolumeDataset(df[df.key.isin(SANITY_KEYS)], transform=paired_tf)
loader   = DataLoader(train_ds, batch_size=1, shuffle=True)
model = build_model().to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200
hist = []
for ep in range(EPOCHS):
    model.train()
    for batch in loader:
        ap, lat = batch["ap"].to(DEVICE), batch["lat"].to(DEVICE)
        gt = batch["gt"].to(DEVICE)
        opt.zero_grad(); out = model(ap, lat)
        loss = total_loss_mc(out, gt); loss.backward(); opt.step()
    if ep % 10 == 0 or ep == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            ds = []
            for batch in loader:
                o = model(batch["ap"].to(DEVICE), batch["lat"].to(DEVICE))
                o = o[0] if isinstance(o, (tuple, list)) else o
                ds.append(per_bone_dice(o, batch["gt"].to(DEVICE)))
            d = np.mean(ds, 0)
        hist.append((ep, float(loss), *d))
        print(f"ep{ep:3d} loss{loss:.3f} dice femur{d[0]:.2f} tibia{d[1]:.2f} patella{d[2]:.2f} fibula{d[3]:.2f}")
```

- [ ] **Step 3: Assert the success criterion**

```python
final = np.array(hist[-1][2:])                 # last per-bone dice
print("final per-bone dice:", dict(zip(BONES, final.round(3))))
assert (final >= 0.90).all(), f"not all bones learned: {dict(zip(BONES, final.round(3)))}"
print("SANITY PASS: encoder-decoder can represent all 4 bones")
```
Expected: all 4 ≥ 0.90. If patella (smallest, hardest) lags, train longer / raise its Dice weight before concluding — record what was needed. If a bone is stuck near 0, that points back to a GT-alignment problem (revisit Task 4) or a dead channel, not model capacity.

- [ ] **Step 4: Plot per-bone Dice curves + predicted-vs-GT slice montage**

```python
H = np.array([h[2:] for h in hist]); E = [h[0] for h in hist]
plt.figure(figsize=(6,4))
for j,b in enumerate(BONES): plt.plot(E, H[:,j], label=b)
plt.xlabel("epoch"); plt.ylabel("Dice"); plt.legend(); plt.title("Per-bone overfit Dice"); plt.show()
# montage: for one case, show GT vs pred mid-slice per bone (sigmoid>0.5)
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/modeling/gt_per_bone_modelling.ipynb
git commit -m "feat: per-bone loss/metrics + learnability sanity-check (overfit)"
```

---

## Task 7: 3D comparison viewer (predicted knee vs GT STLs)

**Files:**
- Modify: `notebooks/modeling/gt_per_bone_modelling.ipynb`

- [ ] **Step 1: Predicted channels → colored per-bone meshes**

```python
from skimage.measure import marching_cubes
BONE_COLORS = {"femur":[230,180,80],"tibia":[120,200,120],"patella":[200,120,200],"fibula":[120,160,230]}

def pred_meshes(model, batch, thr=0.5):
    model.eval()
    with torch.no_grad():
        o = model(batch["ap"].to(DEVICE), batch["lat"].to(DEVICE))
        o = o[0] if isinstance(o, (tuple, list)) else o
        prob = torch.sigmoid(o.float())[0].cpu().numpy()        # (4,T,T,T)
    meshes = {}
    for j, b in enumerate(BONES):
        vol = prob[j]
        if vol.max() < thr:   # nothing predicted
            continue
        verts, faces, _, _ = marching_cubes(vol, level=thr)
        m = trimesh.Trimesh(verts, faces); m.visual.face_colors = BONE_COLORS[b] + [255]
        meshes[b] = m
    return meshes
```

- [ ] **Step 2: Load the matching GT STLs (already in world space) for side-by-side**

```python
def gt_meshes(key):
    out = {}
    for b in BONES:
        stl = case_stl[key].get(b)
        if stl: 
            m = trimesh.load(stl, process=False); m.visual.face_colors = BONE_COLORS[b] + [255]
            out[b] = m
    return out
```
Note: predicted meshes are in **voxel-index space**; GT STLs are in **world mm**. For a true overlay, transform predicted verts by `predrr_img.affine` back to world before comparing, OR transform GT STLs by `inv(affine)` into index space (cheaper — reuse the Task 2 transform). Pick index space for the viewer so both align with the predicted volume.

- [ ] **Step 3: Per-bone surface metrics (mesh-to-mesh) + Dice table**

```python
def surface_stats(pred_m, gt_m, n=2000):
    pg = pred_m.sample(n); gt = gt_m.sample(n)
    _, d_pg, _ = trimesh.proximity.closest_point(gt_m, pg)
    _, d_gp, _ = trimesh.proximity.closest_point(pred_m, gt)
    return dict(chamfer=float(d_pg.mean()+d_gp.mean()),
                hd95=float(np.percentile(np.concatenate([d_pg,d_gp]),95)),
                assd=float((d_pg.mean()+d_gp.mean())/2))
```
Build a per-bone table combining the volumetric `per_bone_dice` (from Task 6) with `surface_stats`. Print as a DataFrame.

- [ ] **Step 4: Render predicted full knee vs GT (validation cell)**

Composite predicted meshes into one scene and the GT meshes into another; export `.glb`/`.obj` or show via `trimesh.Scene(...).show()` / matplotlib 3D. Assert the predicted knee is non-empty:
```python
pm = pred_meshes(model, next(iter(loader)))
assert len(pm) >= 1, "no predicted bone meshes — check threshold/training"
print("predicted bones:", list(pm))
trimesh.Scene(list(pm.values())).export(ROOT / "reports/pred_knee_sanity.glb")
print("wrote reports/pred_knee_sanity.glb")
```

- [ ] **Step 5: Commit**

```bash
git add notebooks/modeling/gt_per_bone_modelling.ipynb
git commit -m "feat: 3D predicted-knee vs GT-STL comparison viewer + surface metrics"
```

---

## Notes for the implementer

- Run every cell with `./.venv/Scripts/python.exe` as the kernel — the default `python` has a broken numpy/matplotlib ABI (project memory).
- DRR/predrr axis conventions: AP/LAT MIP axes in Task 4 Step 3 and the DRR view-name globbing in Task 4 Step 1 may need adjusting to your actual `data/interim/DRRs/fractured` filenames and predrr lift-axis convention (predrr axes L-R / A-P / S-I; AP→axis1, LAT→axis0 per project memory). Verify against one real file before batch use.
- Whether to commit the generated `.nii.gz` volumes or gitignore them is a user decision (Task 3 Step 4 note) — default to committing only CSVs/PNGs.
- This plan delivers the **sanity-check**, not generalization. Full k-fold training, both front-end regimes, and effect-size decoder comparison remain a later stage (out of scope per spec).
```
