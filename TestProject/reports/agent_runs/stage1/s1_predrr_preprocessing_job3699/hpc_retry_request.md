# HPC result request: s1_predrr_z046_right_retry

Please upload the corrected notebook twin and run the targeted retry through Open OnDemand.

- Local notebook to upload: `TestProject/HPC/HPC_notebooks/pre_processing_HPC/predrr_preprocessing.ipynb`
- HPC notebook path: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/HPC/HPC_notebooks/pre_processing_HPC/predrr_preprocessing.ipynb`
- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Environment: the same Python 3.12 `.venv` used for Job 3699.

## Configuration to verify

In the tagged environment-configuration cell:

```python
TARGET_SIZE = 256
RESAMPLE_SPACING = 0.5
ORIENTATION = "LPS"
FOV_MM = 200.0
PROCESS_ONLY_IDS = ["VSD_z046_Right"]
```

In the orientation cell:

```python
"VSD_z046_Right": True
"VSD_016_Left": False
"Case14_PartRight": False
```

If the batch cell prints that more than one volume will be processed, stop the job and correct `PROCESS_ONLY_IDS`.

## Cells to run

1. Restart the kernel.
2. Run from the top through the orientation preview.
3. Confirm the preview processes only `VSD_z046_Right` and the after coronal/sagittal panels show the femur at the top.
4. Run the batch cell. It must report 1/1 processed and metadata rows=71.
5. Run saved-volume shape/spacing validation.
6. Run the all-planes visual-inspection cell; in targeted mode it should render only `VSD_z046_Right`.
7. The heavy 10-case body-envelope before/after replay does not need repeating; Job 3699 evidence is preserved and accepted as an artifact-removal audit.

## Expected updated outputs

- `data/interim/predrr_lps_256_v1/healthy/VSD_z046_Right.nii.gz`
- `data/interim/predrr_lps_256_v1/preprocessing_metadata.csv`
- Executed targeted notebook or exported output log.
- Coronal and sagittal QA figure for the corrected saved volume.
- Run-configuration JSON recording the values above.

## PASS indicators

- Exactly 1/1 requested volume processed.
- Metadata remains 71 unique rows.
- `VSD_z046_Right` records `si_flipped=True`.
- Output is LPS, 256 cubed and 0.78125 mm isotropic.
- Femur is superior/top in both coronal and sagittal QA.
- No traceback.
- The corrected NIfTI path, size and SHA-256 are returned.

## Failure indicators

- More than one volume starts processing.
- Metadata is reduced to one row or contains duplicate IDs.
- `si_flipped` remains false.
- Femur remains below the tibia.
- Shape, spacing or LPS checks change.
- The result bundle omits the corrected-file hash or QA figure.

After the run, return the executed notebook or log, run-configuration JSON, merged metadata CSV, QA figure, traceback if any, job ID, exact GPU model, requested host RAM, wall time, peak GPU and host memory where available, and the corrected NIfTI path/size/SHA-256. The retry remains pending until Agent N reviews that bundle.
