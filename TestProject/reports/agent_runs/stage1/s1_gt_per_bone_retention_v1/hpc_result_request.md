# HPC result request: s1_gt_per_bone_retention_v1

Please upload and run `TestProject/HPC/HPC_notebooks/modeling_HPC/00_gt_per_bone.ipynb` through Open OnDemand.

- HPC notebook path: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/notebooks/00_gt_per_bone.ipynb`
- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Environment: `.venv`, Python 3.12; the audit is CPU/host-memory heavy and does not require a GPU, though the normal GPU queue is acceptable.
- Verify configuration: `RETENTION_PROCESS_ONLY_IDS = None`, `RETENTION_FLOOR = 0.995`, `REMOVED_COMPONENT_FAIL_VOXELS = 30`, and `RETENTION_VERSION = "pb_retention_lps_0p5_v1"`.
- Audit-only execution: restart the kernel; execute the setup/helper cells through `build_bone_gt`, the VSD discovery/helper cells, then `retention-config-v1` and `retention-run-v1`. Do not execute the fractured/VSD target-generation loops or the quarantined TSDF raw cell; the certification cell reads but never overwrites approved masks.
- Expected output root: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/data/interim/gt_per_bone_lps_256_v1/retention_audit_v1/`.

## Expected outputs

- `gt_per_bone_retention.csv`
- `gt_per_bone_retention_summary.json`
- `run_configuration.json`
- `evidence_sha256.txt`
- `qa/<case>_retention_overlay.png` for the eight focused fractured cases, plus any automatic failures

## Automated pass indicators

- `sample_count = 71` and `row_count = 284`.
- `automated_verdict = PASS` and `automated_failure_rows = 0`.
- `minimum_retention_fraction >= 0.995`.
- `maximum_removed_component_voxels < 30`.
- Zero replay, binary/non-empty or predrr-geometry failures.
- Eight named focused QA figures are present.
- Final notebook output prints `AUTOMATED RETENTION GATE PASS`.

## Failure indicators

- Any traceback or final assertion failure.
- Missing/duplicate samples or fewer than 284 rows.
- `automated_verdict = RETRY`, any non-zero failure count, or missing evidence files.
- Any red overlay involving important fractured bone anatomy, even if the numeric thresholds pass.

After the run, please return:

1. the executed notebook or exported cell output/error log;
2. `run_configuration.json`;
3. the retention CSV and summary JSON;
4. all focused QA overlays;
5. `evidence_sha256.txt`;
6. any traceback;
7. GPU model, requested host RAM and wall time;
8. observed peak GPU and host memory where available;
9. job ID and bulk-output path.

Copy the compact bundle to `TestProject/HPC/HPC_results/s1_gt_per_bone_retention_v1/<run_id>/`. The task remains pending until the returned evidence and user visual review receive an Agent N verdict.

