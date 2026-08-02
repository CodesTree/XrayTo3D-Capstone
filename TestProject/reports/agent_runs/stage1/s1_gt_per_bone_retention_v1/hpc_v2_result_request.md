# HPC result request: s1_gt_per_bone_retention_v2

Please upload and run the corrected `TestProject/HPC/HPC_notebooks/modeling_HPC/00_gt_per_bone.ipynb` through Open OnDemand.

- HPC notebook path: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/notebooks/00_gt_per_bone.ipynb`
- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Environment: `.venv`, Python 3.12; CPU/host-memory heavy, GPU optional.
- Verify cell IDs: `retention-contract-v2`, `retention-config-v2`, `retention-run-v2`.
- Verify configuration: `RETENTION_PROCESS_ONLY_IDS = None`, `RETENTION_VERSION = "pb_retention_lps_0p5_v2"`, retention floor `0.995`, significant component threshold `30`.
- Audit-only execution: restart the kernel; execute the setup/helper cells through `build_bone_gt`, the VSD discovery/helper cells, then the three v2 retention cells. Skip the fractured/VSD target-generation loops and quarantined TSDF raw cell.
- Version 1 must remain untouched.
- Expected output root: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/data/interim/gt_per_bone_lps_256_v1/retention_audit_v2/`.

## Expected outputs

- `gt_per_bone_retention_v2.csv`
- `gt_per_bone_retention_summary_v2.json`
- `run_configuration_v2.json`
- `evidence_sha256.txt`
- `qa/<case>_retention_v2_overlay.png` for every fractured case requiring FOV review and every automatic failure

## Automated bone-gate pass indicators

- `sample_count = 71` and `row_count = 284`.
- `roi_crop_failure_rows = 0`.
- `unexpected_fov_failure_rows = 0`.
- Zero replay, binary/non-empty and predrr-geometry failures.
- `automated_bone_failure_rows = 0` and `automated_bone_verdict = PASS`.
- Expected shaft truncation is reported as `expected_boundary_connected_long_bone_shaft`, not hidden.
- Final notebook output prints `AUTOMATED BONE RETENTION V2 PASS`.
- `stage1_target_verdict` must remain `PENDING_FRACTURE_ROI_AND_USER_REVIEW`; this is expected and is not an automated bone-gate failure.

## Failure indicators

- Any traceback or final automated assertion failure.
- Missing/duplicate samples or fewer than 284 rows.
- Any ROI-crop failure.
- Any large FOV component that is disconnected from the retained shaft, affects a non-long-bone outside tolerance, or requires clipping outside the S-I axis.
- Replay, binary/non-empty or predrr-grid mismatch.
- Missing v2 evidence or any modified/missing v1 evidence.

After the run, please return:

1. the executed notebook or exported output/error log;
2. `run_configuration_v2.json`;
3. the v2 retention CSV and summary JSON;
4. every v2 QA overlay;
5. `evidence_sha256.txt`;
6. any traceback;
7. GPU model, requested host RAM and wall time;
8. observed peak GPU and host memory where available;
9. job ID and bulk-output path.

Copy compact evidence to `TestProject/HPC/HPC_results/s1_gt_per_bone_retention_v2/<run_id>/`. The task remains pending until the returned v2 evidence and user visual review receive a gate verdict.
