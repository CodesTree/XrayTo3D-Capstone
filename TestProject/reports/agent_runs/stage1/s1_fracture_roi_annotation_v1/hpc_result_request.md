# HPC result request: s1_fracture_roi_annotation_v1

## Before execution

1. Use working directory `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/` and the project `.venv`.
2. Upload the implemented notebook to the HPC `notebooks/` tree.
3. Upload the edited `reports/manifests/fracture_roi_status_v1.csv`.
4. For every `verified_fracture` row, upload the Slicer-exported binary NIfTI to its declared `source_roi_path` under `data/interim/fracture_roi_lps_256_v1/source_original_ct/`.
5. Confirm the exports use the exact recorded source CT reference geometry. Case3 and Case16 must be annotated on their cast-cleaned NIfTIs.
6. Restart the kernel and execute all cells. This replay is CPU/host-memory work; GPU use is not required.

## Automated pass indicators

- Exactly 13 unique status rows and no `needs_user_review` rows.
- Every verified ROI reports `result=pass` and `pre_resize_retention_fraction=1.0`.
- Zero source-grid, binary, crop, FOV, resize, predrr-grid or bone-intersection failures.
- Every verified ROI has a generated three-plane QA overlay and `visual_approval=approved`.
- Summary reports `PASS_AUTOMATED_READY_FOR_AGENT_N`.

## Failure indicators

- Missing or empty source ROI, non-binary values, or source-grid mismatch.
- Any crop/FOV voxel loss or empty 256-cubed output.
- No intersection with the four-bone target.
- A rejected/pending overlay, `RETRY`, traceback, or incomplete evidence hashes.

## Return bundle

Return the executed notebook or complete log, edited status CSV, run configuration JSON, QC CSV, summary JSON, all QA overlays, SHA-256 evidence manifest, aligned-output path, job ID, host/GPU model where applicable, requested host RAM, wall time and observed peak host/GPU memory. File existence or job completion alone is not a PASS.
