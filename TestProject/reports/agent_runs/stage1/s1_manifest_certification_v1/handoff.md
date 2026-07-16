# Agent B Handoff — Stage 1 Manifest Certification v1

## Task and ownership

- Producer: Agent B — Cohort, Manifest and Leakage Control
- Independent reviewer requested: Agent N
- Scope: deterministically integrate the reviewed fracture ROI record, certify the 71-case quantitative cohort, regenerate manifest evidence, and revalidate leakage.

## Inputs reviewed

- `reports/manifests/fracture_roi_status_v1.csv`: 13 unique included Ruikar cases, all visually approved; 12 `verified_fracture` and Case2 `no_visible_fracture`.
- Returned HPC DRR evidence in `data/interim/DRRs_diffdrr_lps_256_v1/qa_v1_HPC/`: automated PASS, 142 unique QA rows, 147 hash entries, HPC/CUDA configuration.
- Prior Agent N Stage 1 DRR and fracture-target PASS decisions.

## Files changed or regenerated

- `notebooks/data_management/00_manifest_and_reconciliation.ipynb`
- `reports/manifests/quantitative_manifest_v1.csv`
- `reports/manifests/quantitative_manifest_v1.metadata.json`
- `reports/manifests/quantitative_manifest_v1.sha256`
- `reports/manifests/cohort_flow_v1.csv`
- `reports/manifests/leakage_report_v1.json`

## Implementation

- Replaced hard-coded Ruikar `needs_user_review` output with a strict merge from `fracture_roi_status_v1.csv`.
- Requires exact 13-case identity coverage, final ROI status, `visual_approval=approved`, reviewer, and parseable review date.
- Preserves Case2 as `no_visible_fracture`.
- Promotes included rows only when the explicit Stage 1 decision and ROI, HPC DRR, artifact-presence, cohort, and leakage prerequisites pass.
- Records the certification decision, prerequisite results, ROI source hash, fold counts, and expanded split-overlap audit in metadata.
- Notebook was executed twice successfully; outputs were idempotent.

## Verification results

- Total rows: 78
- Ready: 71
- Pending recertification: 0
- Excluded: 7
- Included cohort: 58 VSD healthy + 13 Ruikar fractured
- Fold row counts: 15, 14, 14, 14, 14 for folds 0–4
- Duplicate sample IDs: 0
- Subject assigned to multiple test folds: 0
- Fold 5 rows: 0
- Augmentation-parent mismatches: 0
- Derived train/validation/test subject overlaps: 0 for all five configurations under `validation_fold = (test_fold + 1) mod 5`
- `certification_approved`: `true`
- All recorded certification prerequisites: `true`
- Manifest SHA-256: `bdaef4df5e93f7ff2150051931a024147d4ead8f6f4f80caf1a230a9a590868d`
- SHA-256 independently recomputes and matches the CSV, metadata, and `.sha256` record.
- Notebook JSON is valid and all code cells compile; executed code-cell counts are 1–4.

## Assumptions and remaining gate

- The certification uses the already recorded Agent N PASS decisions and returned evidence; it does not rerender or modify DRRs, ROIs, targets, or source data.
- Real-data Stage 2 remains pending one final Agent N read-only review of this regenerated manifest, metadata, hash, cohort flow, and leakage report.

## Requested verdict

Agent N: independently return PASS, RETRY, or BLOCKED for the regenerated Stage 1 manifest certification and permission to begin real-data Stage 2.
