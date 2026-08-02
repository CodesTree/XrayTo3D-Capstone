# Task brief: s1_predrr_preprocessing_job3699

- Stage: 1
- Producing role: Agent D - Geometry and Projection
- Independent reviewer: Agent N
- Lane: predrr preprocessing and geometry evidence only; no Stage 2 modelling notebooks
- HPC run: Job 3699
- Requested verdict: PASS, RETRY or BLOCKED

## Objective

Review the returned 71-sample `predrr_lps_256_v1` HPC run, explain the fractured-case retention warnings, resolve the reported `VSD_z046_Right` inversion, and prevent dependent target/DRR work from consuming unreviewed geometry.

## Inputs

- Executed `predrr_preprocessing.ipynb`.
- `preprocessing_metadata.csv`.
- `orientation_validation_report.csv`.
- 71 generated NIfTI volumes.
- User screenshots and visual confirmations.
- Job/resource details supplied in the task.

## Success criteria

- Exactly 58 healthy plus 13 fractured outputs match the 71-row manifest.
- Every output is LPS, 256 cubed and 0.78125 mm isotropic.
- Retention QA is interpreted according to its actual computation.
- Every anatomically inverted volume has an explicit, reproducible correction.
- The returned evidence bundle receives an Agent N verdict before dependent work continues.
