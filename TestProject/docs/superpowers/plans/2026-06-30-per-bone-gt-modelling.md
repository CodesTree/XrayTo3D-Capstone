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

## Task 2: Single-case voxelization + visual alignment spot-check

**Files:**
- Modify: `notebooks/modeling/gt_per_bone.ipynb` (add voxelizer + one-case overlay cells)

- [ ] **Step 1: Write the voxelizer function**

```python
def voxelize_bone(stl_path: Path, predrr_img) -> np.ndarray:
    """STL (world RAS mm) -> uint8 occupancy on predrr voxel grid via inv(affine)."""
    inv = np.linalg.inv(predrr_img.affine)
    mesh = trimesh.load(stl_path, process=False)
    v = np.asarray(mesh.vertices)
    vh = np.c_[v, np.ones(len(v))]
    mesh.vertices = (inv @ vh.T).T[:, :3]          # now in voxel-index space
    vg = mesh.voxelized(pitch=1.0).fill()          # solid fill in index space
    idx = np.round(np.asarray(vg.points)).astype(int)
    shape = predrr_img.shape
    mask = np.zeros(shape, dtype=np.uint8)
    inb = (idx >= 0).all(1) & (idx[:, 0] < shape[0]) & (idx[:, 1] < shape[1]) & (idx[:, 2] < shape[2])
    idx = idx[inb]
    mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return mask
```

- [ ] **Step 2: Spot-check one case/bone against the predrr (the visual "test")**

Pick the first mapped case; voxelize its femur; overlay on the predrr mid-slices. This is the alignment smoke test:
```python
key  = sorted(cases)[0]
img  = nib.load(PREDRR / f"{key}.nii.gz")
ct   = np.asarray(img.dataobj)
mask = voxelize_bone(cases[key]["femur"], img)
print(key, "femur voxels:", int(mask.sum()), "ct shape:", ct.shape)
assert mask.sum() > 1000, "femur mask implausibly small — voxelization/fill failed"

# overlay at the slice with the most femur signal, per axis
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for a, axis in enumerate([0, 1, 2]):
    s = mask.sum(axis=tuple(i for i in range(3) if i != axis)).argmax()
    ct_sl   = np.take(ct,   s, axis=axis)
    mask_sl = np.take(mask, s, axis=axis)
    ax[a].imshow(ct_sl.T, cmap="gray", origin="lower")
    ax[a].imshow(np.ma.masked_where(mask_sl.T == 0, mask_sl.T), cmap="autumn", alpha=0.5, origin="lower")
    ax[a].set_title(f"{key} femur axis{axis} slice{s}")
plt.tight_layout(); plt.show()
```

- [ ] **Step 3: Eyeball the overlay**

Expected: the colored femur mask sits **on top of** the bright bone in the CT in all 3 panels. If it is shifted, mirrored, or in empty space, that is a RAS/LPS flip — note it; Task 4 adds systematic flip detection/correction. Do not proceed to batch build if the spot-check is grossly misaligned; instead jump to Task 4's flip-handling and re-run this cell.

- [ ] **Step 4: Commit**

```bash
git add notebooks/modeling/gt_per_bone.ipynb
git commit -m "feat: voxelize STL onto predrr grid with single-case overlay check"
```

---

## Task 3: Batch-build per-bone `.nii.gz` + metadata

**Files:**
- Modify: `notebooks/modeling/gt_per_bone.ipynb`
- Create (output): `data/interim/gt_per_bone_256/fractured/<KEY>/<KEY>_<bone>.nii.gz`, `.../gt_per_bone_metadata.csv`

- [ ] **Step 1: Write the batch-build cell**

```python
import pandas as pd
rows = []
for key, bone_map in sorted(cases.items()):
    img = nib.load(PREDRR / f"{key}.nii.gz")
    case_dir = OUT_ROOT / key
    case_dir.mkdir(parents=True, exist_ok=True)
    for bone in BONES:
        stl = bone_map.get(bone)
        if stl is None:
            mask = np.zeros(img.shape, dtype=np.uint8); src = ""
        else:
            mask = voxelize_bone(stl, img); src = stl.name
        nib.save(nib.Nifti1Image(mask, img.affine, img.header),
                 case_dir / f"{key}_{bone}.nii.gz")
        rows.append(dict(key=key, bone=bone, voxels=int(mask.sum()),
                         source_stl=src, missing=(stl is None)))
meta = pd.DataFrame(rows)
meta.to_csv(OUT_ROOT / "gt_per_bone_metadata.csv", index=False)
meta
```

- [ ] **Step 2: Validate the build (the "test" cell)**

```python
assert (meta.groupby("key").size() == 4).all(), "every case must have 4 bone files"
assert (~meta.missing).all(), f"missing bone STLs:\n{meta[meta.missing]}"
assert (meta.loc[~meta.missing, "voxels"] > 500).all(), \
    f"suspiciously empty masks:\n{meta[(~meta.missing)&(meta.voxels<=500)]}"
n_files = sum(1 for _ in OUT_ROOT.rglob('*.nii.gz'))
assert n_files == 4 * len(cases), f"expected {4*len(cases)} files, found {n_files}"
print("OK:", len(cases), "cases ×4 bones =", n_files, "nii.gz written")
```
Expected: PASS, prints file count. Flag `Case2_PartLeft` (non-fractured) and `Case16_PartRight` (cast-removed source) are still built — they are valid GT, just noted.

- [ ] **Step 3: Manually open one case in 3D Slicer (human gate)**

Load `data/interim/predrr/fractured/<KEY>.nii.gz` + its 4 `gt_per_bone_256/.../<KEY>_<bone>.nii.gz` as segmentations. Confirm each bone overlays the right anatomy. (This is the payoff of choosing `.nii.gz` output.)

- [ ] **Step 4: Commit**

```bash
git add notebooks/modeling/gt_per_bone.ipynb data/interim/gt_per_bone_256/fractured/gt_per_bone_metadata.csv
git commit -m "feat: batch-build per-bone GT nii.gz + metadata for fractured cases"
```
(Note: the `.nii.gz` masks themselves — decide with the user whether to commit binaries or gitignore; default: gitignore the volumes, commit only the metadata CSV.)

---

## Task 4: Alignment validation gate (union-Dice + MIP overlays + flip handling)

**Files:**
- Modify: `notebooks/modeling/gt_per_bone.ipynb`
- Create (output): `data/interim/gt_per_bone_256/fractured/gt_per_bone_alignment.csv`, `.../alignment_grid.png`

- [ ] **Step 1: Locate the existing whole-bone occupancy reference**

```python
OCC = ROOT / "data/interim/predrr_occupancy_256"
GTV5 = ROOT / "data/interim/gt_v5/fractured"
def ref_occupancy(key):
    """Existing single-mask reference for this case, or None if absent."""
    # fractured occupancy npy naming, fall back to gt_v5 nii.gz
    cand = list(OCC.glob(f"*{key.lower()}*.npy")) + list(OCC.glob(f"fractured_{key.lower()}*.npy"))
    if cand:
        return np.load(cand[0]) > 0.5
    g = GTV5 / f"{key}.nii.gz"
    return (np.asarray(nib.load(g).dataobj) > 0) if g.exists() else None
```
Note: fractured occupancy/gt_v5 may not be pre-cached. If `ref_occupancy` returns `None` for all cases, skip the Dice column and rely on the MIP-on-DRR overlay (Step 3) as the gate instead.

- [ ] **Step 2: Compute union-vs-reference Dice**

```python
def dice(a, b):
    a = a.astype(bool); b = b.astype(bool)
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s

ar = []
for key in sorted(cases):
    union = np.zeros(nib.load(PREDRR / f"{key}.nii.gz").shape, bool)
    for bone in BONES:
        union |= np.asarray(nib.load(OUT_ROOT / key / f"{key}_{bone}.nii.gz").dataobj) > 0
    ref = ref_occupancy(key)
    ar.append(dict(key=key, union_voxels=int(union.sum()),
                   dice_vs_ref=(dice(union, ref) if ref is not None else np.nan)))
align = pd.DataFrame(ar); align.to_csv(OUT_ROOT / "gt_per_bone_alignment.csv", index=False)
align
```

- [ ] **Step 3: MIP overlay of GT union on the actual DRR input (the visual gate)**

```python
DRR = ROOT / "data/interim/DRRs/fractured"
def load_drr_png(key, view):                 # view in {'AP','LAT'} — match your DRR naming
    import matplotlib.image as mpimg
    cand = list(DRR.glob(f"*{key}*{view}*")) + list(DRR.glob(f"*{key.lower()}*{view.lower()}*"))
    return mpimg.imread(cand[0]) if cand else None

n = len(cases); fig, ax = plt.subplots(n, 2, figsize=(8, 3.2 * n))
for r, key in enumerate(sorted(cases)):
    union = np.zeros(nib.load(PREDRR / f"{key}.nii.gz").shape, bool)
    for bone in BONES:
        union |= np.asarray(nib.load(OUT_ROOT / key / f"{key}_{bone}.nii.gz").dataobj) > 0
    # AP = project along anterior axis, LAT = along lateral axis (adjust axes to your predrr convention)
    ax[r, 0].imshow(union.max(axis=1).T, origin="lower"); ax[r, 0].set_title(f"{key} GT MIP (AP)")
    ax[r, 1].imshow(union.max(axis=0).T, origin="lower"); ax[r, 1].set_title(f"{key} GT MIP (LAT)")
plt.tight_layout(); plt.savefig(OUT_ROOT / "alignment_grid.png", dpi=110); plt.show()
```

- [ ] **Step 4: Decide pass/fail and handle flips if needed**

```python
THRESH = 0.80
bad = align[align.dice_vs_ref < THRESH].dropna(subset=["dice_vs_ref"])
print("cases below threshold:", list(bad.key) or "none")
```
If cases fail Dice OR look mirrored/shifted in the MIPs, the cause is almost always an axis flip between Slicer-RAS and the predrr storage. Add a corrected voxelizer that flips the suspect axis and re-run Tasks 3–4 for affected cases:
```python
def voxelize_bone_flip(stl_path, predrr_img, flip_axes=()):
    mask = voxelize_bone(stl_path, predrr_img)
    for ax_ in flip_axes:
        mask = np.flip(mask, axis=ax_)
    return mask
```
Empirically find `flip_axes` by re-running the Step-3 overlay until the union lands on the bone; record the chosen `flip_axes` in a markdown cell. Re-run Task 3 batch-build using `voxelize_bone_flip(..., flip_axes=CHOSEN)` so the cache is corrected. **Do not advance to Task 5 until every case passes the visual gate.**

- [ ] **Step 5: Commit**

```bash
git add notebooks/modeling/gt_per_bone.ipynb data/interim/gt_per_bone_256/fractured/gt_per_bone_alignment.csv
git commit -m "feat: alignment validation gate for per-bone GT (Dice + MIP overlays)"
```

---

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

Per-bone loader that stacks the 4 `.nii.gz`:
```python
def load_gt_per_bone(key):
    vols = [np.asarray(nib.load(GT_PB / key / f"{key}_{b}.nii.gz").dataobj) > 0 for b in BONES]
    return torch.from_numpy(np.stack(vols).astype(np.float32))   # (4,T,T,T)
```
Copy `build_paired_index()` and `PairedDRRVolumeDataset` from `decoder_pipeline.ipynb`, then adapt two things: (a) add a `key` column to the index equal to the predrr stem (the same `Case<N>_Part<Side>` string), filtered to `df.key.isin(KEYS)`; (b) in `__getitem__`, replace the GT line with `gt = load_gt_per_bone(r.key)` so `batch["gt"]` is `(B,4,T,T,T)`. Reuse the existing DRR loading (`load_drr`, `paired_tf`) unchanged — inputs are still AP+LAT.

- [ ] **Step 5: Validate shapes (the "test" cell)**

```python
m = build_model().eval()
ap = torch.randn(1, 3, 256, 256); lat = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    out = m(ap, lat)
out0 = out[0] if isinstance(out, (tuple, list)) else out
assert out0.shape[1] == 4, f"expected 4 output channels, got {out0.shape}"
gt = load_gt_per_bone(KEYS[0])
assert gt.shape[0] == 4, gt.shape
print("model out:", tuple(out0.shape), "gt:", tuple(gt.shape))
```
Expected: output channel dim == 4, gt first dim == 4.

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
- This plan delivers the **sanity-check**, not generalization. Full k-fold training, both front-end regimes, and Wilcoxon comparison remain a later stage (out of scope per spec).
```
