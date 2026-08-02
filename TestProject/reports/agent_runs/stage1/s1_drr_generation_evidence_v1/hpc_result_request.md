# HPC result request: s1_drr_generation_evidence_v1

Run only the import, configuration and final `DRR Evidence Bundle v1` cells in the corrected HPC `drr_generation.ipynb`. Do not run the smoke-test or batch-rendering cells and do not modify existing DRRs.

- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Environment: `.venv`, Python 3.12, CUDA available.
- Existing DRR root: `data/interim/DRRs_diffdrr_lps_256_v1/`.
- Expected evidence root: `data/interim/DRRs_diffdrr_lps_256_v1/qa_v1/`.

## Expected outputs

- `drr_numerical_qa_v1.csv`
- `drr_numerical_qa_summary_v1.json`
- `run_configuration_v1.json`
- `resource_usage_v1.json`
- `evidence_sha256.txt`
- Existing `smoke_test_qa.png` and `visual_inspection_grid.png`
- Existing `drr_generation_metadata.csv`

## Pass indicators

- 71 manifest samples and 142 unique views.
- 116 healthy and 26 fractured views; 71 AP and 71 LAT.
- Zero missing, unexpected or duplicate NPY, PNG or metadata keys.
- Zero shape, dtype, finite-value, range, dynamic-range, PNG-correspondence or paired-view failures.
- Exactly 142 NPY model-input hashes plus compact-evidence hashes.
- `automated_drr_verdict = PASS`.
- `stage1_drr_evidence_status = PASS_READY_FOR_AGENT_N` when numerical, inventory, configuration, hash and QA-figure checks pass. Missing job ID, GPU model, requested RAM or wall time is informational and non-blocking for Stage 1.

## Failure indicators

- Any traceback before the compact evidence files are written.
- Any inventory, metadata, geometry, NPY, PNG, paired-view or QA-figure failure.
- Fewer than 142 NPY hashes or any missing compact evidence.
- Any claim of final Stage 1 certification by the producer notebook.

After the audit, return the executed notebook or exported output, all five `qa_v1` files, both QA PNGs, metadata CSV, any traceback, available job/resource details, and the exact bulk-output path. Copy compact evidence to `TestProject/HPC/HPC_results/s1_drr_generation_evidence_v1/<run_id>/`. The task remains pending until Agent N records PASS.
