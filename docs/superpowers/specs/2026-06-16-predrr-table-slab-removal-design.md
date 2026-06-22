# Design: Remove CT table / support slab from predrr volumes

**Date:** 2026-06-16
**Notebook:** `notebooks/pre-processing/predrr_preprocessing.ipynb` (+ HPC twin)

## Problem

The binary bone-occupancy ground truth (and the DRR model inputs derived from the same volumes)
contain a thin, full-height **vertical slab at the lateral edge** of the scan — a CT table / leg
support / reconstruction-FOV edge. In the 3D GT showcase it appears as "lines at the back."

### Evidence (Case12_PartRight, fractured)
- Slab is low density (~200 HU, median normalized intensity 0.435), peripheral (L-R 183–228,
  A-P 25–40; bone centroid is at 145,139), and spans the full S-I height.
- It is **systematic, not fracture-specific** (healthy VSD_001 has a faint one too) → rules out
  cast/hardware.
- Vanishes at threshold ≥ 0.50 across all cases, because the slab sits below cortical bone (0.585).

### Root cause
`roi_bone_crop` finds the largest connected component only to compute a **bounding box**, then crops
the *original windowed volume* — it never **masks out** non-bone structures inside the box. The slab
is saved at full intensity and passes the decoder's `GT_THRESH=0.40` (~150 HU). Fractured (Ruikar)
cases additionally **skip `vsd_knee_cropping.ipynb`**, so they never receive that notebook's
table filtering — which is why the fractured slab is the most prominent.

### Why the existing `_is_table_component` heuristic is NOT reused
Empirically tested: it returns `table?=False` for every component on this data, because (a) at the
bone threshold the slab merges into the main leg component (no separate component to flag), and
(b) the heuristic is geometry-specific to the VSD *posterior horizontal* table, not this *lateral
vertical* slab. It is also only applied during VSD bilateral leg separation.

## Approach: body-envelope mask (validated)

Add `body_envelope_mask(arr_windowed)` and call it between step 4 (bone window) and step 5 (ROI
crop) in both the demo cell and `process_single_volume`. Orientation-agnostic (no axis assumptions):

1. `body = arr_windowed > BODY_SOFT_THRESHOLD` (0.10, soft tissue — matches `ROI_INTENSITY_THRESHOLD`).
2. 3D morphological **closing** (`BODY_CLOSE_ITER=2`) to consolidate the limb; optional **opening**
   (`BODY_OPEN_ITER=0` by default) as a guard to sever thin table↔skin bridges.
3. Label connected components; **keep the largest** (the limb).
4. `binary_fill_holes` so internal bone/marrow is retained.
5. Zero `arr_windowed` outside the kept mask.

The slab is separated from the limb by an air gap (< 0.10), so it forms a separate component and is
dropped. Soft tissue is preserved → DRR realism unchanged.

### Validation (measured)
| Case | Bone retained (>0.40) | Peripheral full-height columns |
|---|---|---|
| Case12 (fractured) | 97.5% | 36 → 0 (slab removed) |
| VSD_001 (healthy) | 99.5% | shaft columns correctly kept |

### Fracture-safety guard
Displaced fragments live inside the skin envelope, so they are kept. `process_single_volume` records
`body_retention` (fraction of bone>0.40 surviving) in the metadata CSV. A QA cell:
- **Over-removal guard:** flag any case with `body_retention < RETENTION_FLAG` (0.95) for manual review.
- **Under-removal guard:** recompute peripheral full-height column count on saved volumes; flag any
  case where the slab survived.

## Scope & cost
- Files: `predrr_preprocessing.ipynb` (local) + HPC twin. CONFIG constants, one helper, two call
  sites, one QA cell.
- Changes saved volumes → **regenerate predrr (72 files) → DRRs → GT occupancy cache**.
- Cleans both the GT label and the DRR inputs at the source.

## Out of scope
- Decoder-side threshold changes (rejected: leaves slab in DRR inputs, harms fracture detail).
- Reworking `vsd_knee_cropping` / `_is_table_component`.
