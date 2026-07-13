# Handoff: s1_predrr_preprocessing_job3699

- Stage: 1
- Producing roles: Project Orchestrator and Agent D
- Independent reviewer: Agent N
- HPC subtask verdict: PASS after the targeted `VSD_z046_Right` retry
- Full Stage 1 status: BLOCKED pending certified targets, DRRs and fracture ROIs

## Evidence that passed

- Job 3699 produced 71 NIfTIs: 58 healthy and 13 fractured, with an exact match to the included manifest IDs.
- All 71 local returned files are float32, LPS, 256 cubed and 0.78125 mm isotropic.
- The executed notebook has no error outputs and reports 71/71 processed.
- Output geometry uses the locked 0.5 mm intermediate grid, 200 mm FOV and 0.78125 mm final spacing.
- The residual full-height peripheral-column check is 0/71.
- User-confirmed upright controls remain no-flip: `Case14_PartRight` and `VSD_016_Left`.
- The targeted retry processed 1/1 requested volume, retained a 71-row unique metadata table and produced no traceback.
- The corrected `VSD_z046_Right` file is LPS, float32, 256 cubed and 0.78125 mm isotropic; the user confirmed upright anatomy in 3D Slicer.

## Issue 1 resolution: retention warning

`envelope_retention` is a high-HU foreground-retention proxy, not per-bone retention. It counts voxels above normalized 0.40, equivalent to 150 HU, before and after keeping the largest soft-tissue envelope. The denominator includes bone, table, casts and supports.

Eight of 13 fractured cases are below 0.95, while zero of 58 healthy cases are. This is dataset-specific artifact-removal behaviour and remains a review warning, not an anatomical failure or exclusion rule.

A local replay reproduced `Case2_PartLeft = 0.608303` exactly:

- retained limb: 2,222,706 high-HU voxels;
- removed support component 1: 682,062;
- removed support component 2: 746,316.

The two support components explain about 99.8 percent of removed high-HU voxels. Case2 passes the body-envelope cleaning audit. True anatomy retention will be certified later from the transformed femur, tibia, patella and fibula masks after the complete FOV transform.

## Issue 2 resolution: VSD_z046_Right

The original Job 3699 file had a formal LPS header but inverted voxel content. The notebook twins now pin:

`"VSD_z046_Right": True`

The targeted retry records `si_flipped=True`. The corrected NIfTI SHA-256 is `35b14a4587fc0ad5e900a9ef42d4539263364d4eda5f503e1651184df9e87c23`; independent header inspection confirms the locked geometry, and the user confirmed femur-superior anatomy in coronal and sagittal 3D Slicer views. This transform must be replayed byte-identically for the four-channel target and fracture ROI.

## Implemented retry support

- Preserved the original executed Job 3699 notebook and CSV evidence under `HPC/HPC_results/predrr_preprocessing/job_3699/`.
- Added the z046 Right manual orientation pin to local and HPC notebook twins.
- Reclassified the 0.95 value in notebook text as a high-HU artifact-removal proxy.
- Added `PROCESS_ONLY_IDS` for a one-case retry.
- Targeted successful rows merge into the existing 71-row metadata table instead of replacing it.
- Targeted orientation preview and post-save gallery restrict themselves to the requested IDs.
- Preserved the historical Job 3699 `RETRY` bundle and archived corrected retry evidence separately under `HPC/HPC_results/predrr_preprocessing/z046_retry_20260714/`.

## Verification

- Both notebooks parse as JSON and every code cell compiles.
- All non-environment cell sources are identical between local and HPC twins.
- Non-environment source SHA-256: `42f1ed14b9274b7f6526f4b40c3ff755a64cae81b035ed86b3dc04a60c661fd1`.
- Synthetic targeted selection and 71-row metadata merge test passes.
- Corrected metadata contains 71 rows and 71 unique IDs; its only field-level difference from the historical copy is `VSD_z046_Right.si_flipped: False -> True`.
- Agent N independently approved the predrr preprocessing subtask.

## Remaining evidence gaps

- Exact GPU model and requested host RAM were not returned.
- The reported 8.63 GiB peak-memory type is not identified as GPU or host memory.
- No notebook-generated run-configuration JSON was returned.
- No HPC SHA-256 manifest for the 71 bulk NIfTIs was returned.
- A separate job ID and resource record for the one-case retry were not returned; this is recorded as an administrative, non-blocking gap.

The predrr dependency may proceed. Full Stage 1 remains BLOCKED until the four-channel targets, fracture ROIs and AP/LAT DRRs are certified and receive the final Agent N Gate 1 review.
