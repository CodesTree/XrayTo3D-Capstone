# Cross-agent note: fracture-ROI annotation progress + on-disk data changes

- Stage: 1 (Codex / Agent C owns this lane)
- Author of this note: Claude Code, assisting the user during manual annotation
- Date: 2026-07-15
- Scope: **data/input files only** — the notebook `01_fracture_roi_annotation.ipynb` code was **NOT modified**. All validation below was run with the notebook's own helpers (cells 0–2) via the project venv.

## Status change since your `handoff.md`

Your baseline was `PENDING_USER_ANNOTATION` (13 `needs_user_review`, 0 automated failures).
The user has since annotated in 3D Slicer and edited `fracture_roi_status_v1.csv`:

- `roi_status`: **12 `verified_fracture`** + **1 `no_visible_fracture`** (`Case2_PartLeft`)
- `reviewer` = `author`, `review_date` = `15-7-2026` (one row has a leading space; `.strip()` guard tolerates it)
- `visual_approval` = **`pending` on all 13** (see "Remaining gates")

## Current replay standing (re-verified against the locked guards)

**All 12 verified cases now pass every guard** (grid match, binary, 100% crop+FOV retention,
non-empty aligned mask, bone intersection): Case11, 12, 13, 14, 15, 16, 1, 3, 5, 6, 7, 9.

Case12/15/3 previously failed the strict `fov_count == pre_count` (retention 99.99–99.999%): a
handful of ROI voxels fell outside the 200 mm FOV. Diagnosis showed these were **isolated
disconnected specks** at the top/bottom of the segmentation (S-I axis), 6–39 mm outside the FOV —
stray paint, not fracture, and none attached to the main ROI blob. They were removed
programmatically (see data change 4). Verified that removing them left the 256³ `aligned` mask and
the bone-intersection count **unchanged** for all three — i.e. zero usable fracture content was lost;
those voxels were outside the reconstruction volume and could never have appeared in the output.

## On-disk data changes made (so your next run reconciles cleanly)

1. **All 12 masks compressed `.nii` → `.nii.gz`** in `data/interim/fracture_roi_lps_256_v1/source_original_ct/`.
   The notebook's `source_roi_path` expects `.nii.gz`; the user had exported uncompressed `.nii`,
   which failed `assert source_roi_path.exists()`. Compression is lossless — voxel counts are
   byte-identical before/after (cross-checked with raw gunzip, bypassing ITK).
2. **Raw uncompressed `.nii` moved** to `data/interim/fracture_roi_lps_256_v1/_source_raw_nii_uncompressed/`
   (moved, not deleted). Safe to delete to reclaim ~1.7 GB once the bundle is accepted.
3. **Case9 mask rebuilt** onto the correct grid — see below.
4. **Case12 / Case15 / Case3 trimmed** — removed only the ROI voxels landing outside the 200 mm
   FOV (9 / 13 / 4 source voxels respectively). This is the minimal change that satisfies the
   locked `retention == 1.0` guard, and it removed zero reconstructable content (aligned + bone
   counts identical before/after). Pre-trim originals backed up to
   `data/interim/fracture_roi_lps_256_v1/_pretrim_backup/`.
   NOTE for QA: the raw masks are noisy — Case12 had 60 disconnected components, Case3 had 31,
   Case15 had 8. Only the out-of-FOV specks were removed; in-FOV specks (if any) were left as-is.

## ⚠️ GOTCHA that will bite the notebook: `.nii` shadows `.nii.gz` in ITK

With both `X.nii` and `X.nii.gz` present in the same folder, `sitk.ReadImage("X.nii.gz")` silently
pulls **pixel data from the uncompressed `X.nii`** (keeps the `.nii.gz` header/grid, uses the
`.nii` voxels) — no error raised. This produced a bogus 27% retention for Case9 until the `.nii`
siblings were moved out. **Keep only `.nii.gz` in `source_original_ct/`.** Verify reads
independently with: `gzip -dc file.nii.gz | tail -c +353 | tr -d '\000' | wc -c` (nonzero voxel bytes).

## Case9 source-volume mismatch (resolved, no re-segmentation)

- The user segmented Case9 on a **0.7 mm / 732-slice** reconstruction (512×512×732 @ 0.367/0.7 mm).
- The recorded pipeline source (`preprocessing_metadata.csv`) is the **3.0 mm / 91-slice DICOM**
  at `data/raw/fractured/PartRight/Case9` (512×512×91 @ 0.451/3.0 mm) — the series the bone targets
  and DRRs were built from. Grids differ → `images_share_grid` failed.
- The two volumes share the **same physical patient frame** (identical direction; fracture region
  fully inside the 3.0 mm extent), so the 0.7 mm mask was **resampled onto the recorded 3.0 mm grid**
  (nearest-neighbour, physical-space, `Transform()=identity`). Result: 96,324 voxels, grid matches
  the recorded CT, **100% replay retention, 113,129-voxel bone intersection** — passes all guards.
- The hand-drawn 0.7 mm mask is preserved in `_source_raw_nii_uncompressed/Case9_PartRight_fracture_roi_source.nii`.
- Caveat: the mask is now quantised to 3.0 mm Z resolution. This is inherent to using the recorded
  3.0 mm source and is consistent with the rest of Case9's pipeline (targets/DRRs). If Codex/user
  ever decides the 0.7 mm series should be canonical for Case9, that is a larger change (re-run
  Case9 preprocessing end-to-end) — flag before doing it.

## Remaining gates before Stage 1 fracture-ROI PASS

1. ~~Fix Case12 / Case15 / Case3 FOV clipping.~~ **DONE** (data change 4). All 12 replays now pass.
2. `visual_approval` is `pending` on all rows — even though all replays pass, verdict logic returns
   `PENDING_USER_VISUAL_APPROVAL` until the user reviews the QA overlays and sets `approved`.
   **This is the current blocking step.**
3. Independent Agent N fracture-target review (unchanged from your handoff).
4. AP/LAT DRR certification remains separate and pending (unchanged).

## Files touched by this session

- `data/interim/fracture_roi_lps_256_v1/source_original_ct/*.nii.gz` — 12 canonical masks (Case9 = resampled; Case12/15/3 = FOV-trimmed)
- `data/interim/fracture_roi_lps_256_v1/_source_raw_nii_uncompressed/*.nii` — 12 raw uncompressed backups (deletable)
- `data/interim/fracture_roi_lps_256_v1/_pretrim_backup/*.nii.gz` — Case12/15/3 pre-trim originals (deletable)
- `reports/manifests/fracture_roi_status_v1.csv` — user-edited annotation state
- (notebook code unchanged; QA overlays/QC CSV/summary regenerate on the next notebook run)
