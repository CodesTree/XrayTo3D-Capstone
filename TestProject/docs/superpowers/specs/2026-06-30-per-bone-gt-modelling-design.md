# Per-Bone Ground-Truth Modelling — Design Spec

**Date:** 2026-06-30
**Author:** Chan Zheng Shao (with Claude Code)
**Status:** Approved design, pending implementation plan

## Goal

Replace the single whole-bone occupancy target with **manually-segmented per-bone
ground truth** (femur, tibia, patella, fibula) exported as STL from 3D Slicer, and verify
that the existing shared encoder + U-Net/V-Net decoder can **learn and predict each bone
independently**. Immediate objective is a **learnability sanity-check**: prove model capacity
on a few cases before committing to full training.

## Decisions (locked)

| Decision | Choice |
| --- | --- |
| Architecture | **One multi-label model**: shared encoder/fusion unchanged, decoder head outputs **4 sigmoid channels**, channels may overlap. |
| Channel order | `[femur, tibia, patella, fibula]` (canonical). |
| STL source frame | Segmented on **original raw CT** → STL world coords are raw-CT physical frame. |
| Alignment | Voxelize onto each case's **predrr affine** (`world→voxel`), since resample/reorient/crop preserve physical space. RAS/LPS flip is the main risk → mandatory overlay validation. |
| GT cache format | **Per-bone `.nii.gz`** (4 files/case) so they load into Slicer beside the originals; dataset loader stacks to 4-ch tensor at load time. |
| Scope | Start with the **11 ready fractured cases**; VSD GT only started (VSD_002). |
| Voxelization lib | **trimesh** (new dependency). |

## Existing pipeline (context)

- `notebooks/modeling/decoder_pipeline.ipynb`: `AP+LAT DRRs → shared encoder/fusion →
  3D features [64,128,256,512] → U-Net/V-Net decoder → binary occupancy`.
- Current GT: CT bone-windowed, thresholded whole-bone occupancy (`data/interim/gt_v5`,
  `data/interim/predrr_occupancy_256`).
- `notebooks/modeling/decoder_ui.ipynb`: renders predicted volume as 3D mesh
  (`gr.Model3D`, `volume_to_obj`) + slice views — basis for the comparison viewer.
- predrr volumes (affine source): `data/interim/predrr/fractured/Case{N}_Part{Left,Right}.nii.gz`,
  `data/interim/predrr/healthy/VSD_xxx_{Left,Right}.nii.gz`. Target grid 256³.

## Components

### Component 1 — STL → per-bone voxel GT

New notebook `notebooks/modeling/gt_per_bone.ipynb`.

- **Filename parser**: map each STL → `bone ∈ {femur,tibia,patella,fibula}` by case-insensitive
  substring match, **ignoring** noise tokens (`left`, `right`, `segmentation`, `solid`, `_`,
  digits, `cast_removal`). Handles all observed variants:
  `Case1_tibia.stl`, `Case11 segmentation_left femur.stl`, `Segmentation_Fibula.stl`,
  `case_16_cast_removal segmentation_1_solid_tibia.stl`, `VSD_002_Right segmentation_left femur.stl`.
- **Case→predrr key**: `Case{N}/` → `Case{N}_Part{Left|Right}` (side read from predrr filename,
  not the STL token); `VSD_002/Left` → `VSD_002_Left`. `Case2 (does not seem fractured)` →
  `Case2_PartLeft` (flagged in metadata).
- **Voxelization**: load case predrr for `affine` + `shape`; transform mesh vertices by
  `inv(affine)` into voxel-index space; trimesh voxelize at `pitch=1` + solid-fill; scatter into
  `(256,256,256)` `uint8`.
- **Output**: `data/interim/gt_per_bone_256/{fractured,healthy}/<key>/<key>_{femur,tibia,patella,fibula}.nii.gz`,
  each written with the **predrr affine + header** so it overlays the predrr in Slicer.
- **Edge cases**: missing bone STL → empty volume + warning row; ambiguous/duplicate bone match →
  fail loudly (do not guess). Metadata CSV `gt_per_bone_metadata.csv`: case, side, bone, n_vertices,
  voxel_count, source_stl, flags.

### Component 2 — Alignment validation (mandatory gate)

- Per case: union the 4 bone volumes → occupancy; compute **Dice vs existing
  `gt_v5`/`predrr_occupancy`** mask; render **AP + LAT MIP overlays** of voxelized GT on the actual
  DRR input.
- Low union-Dice or visibly shifted overlay ⇒ suspected RAS/LPS flip → detect + correct via axis
  permutation/flip, re-validate.
- **Output**: validation grid PNG + `gt_per_bone_alignment.csv` (per-case union-Dice).
- **Gate**: do not proceed to Component 3 until overlays register visually and union-Dice is high
  (target ≥ ~0.8 vs existing occupancy, acknowledging GT-definition differences).

### Component 3 — Multi-label head + learnability sanity-check

- Decoder final conv `out_channels 1→4`, **sigmoid**; loss = mean over channels of
  `(BCE + soft-Dice)`. Encoder/fusion unchanged. Metrics reported **per channel**.
- **Sanity-check protocol**: pick ~3 fractured cases; `train == val == those cases`; no augmentation;
  overfit many epochs.
- **Success criterion**: **per-bone val Dice ≥ 0.90 on all 4 channels**. Deliver per-bone Dice
  curves + predicted-vs-GT slice montage. (Proves capacity, not generalization.)

### Component 4 — 3D comparison viewer

- Extend `decoder_ui.ipynb` pattern: each predicted channel → marching cubes → colored per-bone mesh
  → composited full-knee model; rendered beside the **original GT STLs** (same world frame).
- Report per-bone **Dice** + **surface metrics** (Chamfer / Hausdorff / mean surface distance),
  mesh-to-mesh.
- Caveat surfaced in UI: at sanity-check stage inputs are overfit training cases; viewer is reused
  for held-out cases in the later full-training stage.

## Dependencies

- Add **`trimesh`** (+ optional `rtree` for faster inside-tests) to the env / `requirements`.
  Flag as the only new dependency.

## Out of scope (this spec)

- Full k-fold training, both front-end regimes, Wilcoxon comparison (later stage).
- Building additional VSD ground-truth segmentations.
- Modifying Lai's reference notebooks `01–10` (read-only).

## Success criteria (overall)

1. Per-bone `.nii.gz` GT generated for all 11 fractured cases, each overlaying its predrr in Slicer.
2. Alignment validation passes (overlays register; flips corrected).
3. Sanity-check: all 4 bone channels reach val Dice ≥ 0.90 on the overfit set.
4. 3D viewer shows predicted full knee vs GT STLs with per-bone Dice + surface distances.

---

## Revision 2026-06-30 — Alignment via pipeline-replay (supersedes Components 1–2 alignment)

**Discovery during Task 2:** the predrr `.nii.gz` volumes were written with a **zeroed-translation
affine** (origin at world 0,0,0; correct 0.78125 mm spacing + RAS-ish direction only). The predrr
build (`predrr_preprocessing.ipynb::process_single_volume`) does its S-I flip, intensity-based ROI
crop, fixed-FOV centering and resize in **numpy array space**, discarding world coordinates; the ROI
crop origin is data-derived and **not stored** in `preprocessing_metadata.csv` (only `crop_shape`).
So the original plan's "voxelize directly onto the predrr affine" is impossible.

**Verified facts (Case11):** the raw fractured CTs still exist
(`data/raw/fractured/Part{Left,Right}/Case<N>/`, DICOM series) and share the STL's world frame —
the STL femur world box (X −93.3..−18.4, Y −121.6..−54.6, Z 1271..1401) sits cleanly **inside** the
raw CT world box (X −158.8..28.8, Y −188.8..−1.2, Z 1162.9..1400.2). The Slicer STL is in the CT's
LPS frame — **no RAS/LPS sign flip** needed.

**Revised alignment approach (replaces Component 1 voxelization + Component 2 gate):**
1. Voxelize each STL onto the **raw CT grid** (raw CT `TransformPhysicalPointToContinuousIndex`),
   producing a per-bone label image with the raw CT's geometry.
2. Push each per-bone label through the **identical geometric transform** that built predrr —
   `resample (NEAREST) → orient RAS → ROI crop (reusing the CT-derived crop box) → center 400³ →
   resize 256³ (NEAREST) → S-I flip (reusing the CT's `correct_orientation` decision)`. The crop
   indices and flip flag are computed from the CT and **shared** with the labels so they land
   pixel-perfect on the predrr/DRR grid.
3. **Keystone validation gate:** replay the *CT* through the same ported functions and confirm it
   reproduces the stored predrr volume (bone-mask Dice ≥ 0.99 / `allclose`). If the CT replay
   matches, the bone labels are aligned **by construction**. Then overlay GT-union on predrr +
   per-case Slicer eyeball as before.

**Implementation note:** the ported transform functions (`resample_volume`, `orient_volume`,
`apply_bone_window`, `body_envelope_mask`, `roi_bone_crop` — refactored to also return crop indices,
`center_to_fixed_fov`, `resize_volume`, `correct_orientation`, `SI_FLIP_OVERRIDE`, constants) are
copied from `predrr_preprocessing.ipynb` into `gt_per_bone.ipynb` (notebook-first; the reference
notebook stays read-only). Output `.nii.gz` is written with the **stored predrr affine + header** so
it overlays predrr in Slicer. The direct-affine `voxelize_bone` added in commit `5f22511` is removed.
