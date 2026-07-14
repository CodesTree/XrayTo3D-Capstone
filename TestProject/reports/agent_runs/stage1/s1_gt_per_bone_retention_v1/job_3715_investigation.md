# Job 3715 investigation: fractured retention failures

- Producing role: Agent C - Fracture and Target Integrity
- Job ID: 3715
- Producer verdict: RETRY
- Meaning: correct the certification method; do not discard the fractured targets.

## Returned evidence integrity

- The executed notebook contains 21 executed code cells.
- The retention cell completed its CSV, JSON, QA and hash outputs before raising the intended gate assertion.
- Exactly 71 samples and 284 sample/bone rows were audited.
- All 16 entries in `evidence_sha256.txt` match the returned files.
- There are zero replay-to-saved mismatches, zero binary/non-empty failures and zero predrr-grid failures.

## Failure pattern

- All 232 healthy sample/bone rows pass.
- In the 13 fractured samples, every femur, tibia and fibula fails: 39 rows total.
- All 13 fractured patellae pass; 12 retain 100 percent and Case13 retains 99.9991 percent with one removed voxel.
- Every failed row has the same two reasons: retention below 0.995 and a removed component of at least 30 voxels.
- In every failed row, the largest component represents at least 99.8185 percent of all removed voxels; the median is 100 percent. This is a single continuous shaft end, not scattered internal fragments.

## Root cause

The audit combines two operations into one failure metric:

1. unintended loss caused by the ROI crop; and
2. intentional exclusion outside the locked 200 mm field of view.

For every fractured scan, the ROI crop preserves the full source superior-inferior length, but that length is greater than the 400-voxel fixed box:

- fractured crop length: 426 to 618 voxels at 0.5 mm, or 213 to 309 mm;
- deliberate total S-I trimming: 13 to 109 mm;
- `clipped_z=True` for all 13 fractured scans and false on the other axes;
- healthy crop length: 397 to 400 voxels, so no comparable post-crop FOV trimming is recorded.

Representative Case1, Case2 and Case14 overlays show the removed component at the remote femoral, tibial or fibular shaft end while the central knee anatomy and patella remain in the field of view. The final masks confirm that all 13 fractured femora, tibiae and fibulae terminate on the appropriate z boundary. Boundary termination is also common in the healthy targets: 54/58 femora, 38/58 tibiae and 32/58 fibulae touch a z boundary because their source acquisitions already end near the 200 mm knee domain.

The 99.5 percent and 30-voxel criteria therefore cannot be applied to the entire longer Ruikar source STL while also keeping a 200 mm physical reconstruction contract. The audit currently labels a deliberate domain definition as preprocessing damage.

## Modelling impact

- There is no input-target geometry mismatch: all 284 saved masks replay exactly and share the predrr grid.
- The model can train on these targets as 200 mm knee-ROI targets; it cannot be described as reconstructing the complete source femur, tibia or fibula.
- Artificial shaft cut planes may inflate global Dice/IoU and influence HD95/ASSD, so fracture-local metrics and boundary-censored surface metrics remain necessary.
- Fracture anatomy is not yet certified. Manual fracture-region masks must be approved in original CT LPS space and replay with 100 percent retention before fracture-local training or metrics are allowed.
- If a fracture ROI intersects the excluded shaft, that case must be recentered, use a larger FOV, or be excluded from fracture-local analysis.

## Required correction

Revise the notebook to report and gate the operations separately:

1. ROI-crop retention: at least 99.5 percent per bone and no unintended removed component of at least 30 voxels.
2. Fixed-FOV truncation: report separately; permit only boundary-contiguous long-bone shaft truncation.
3. Patella and internal-fragment rule: reject any important internal or patellar anatomy removed by either operation.
4. Fracture-ROI rule: require 100 percent retention after the complete transform.
5. Surface metrics: identify/censor artificial volume-boundary surfaces for HD95 and ASSD while retaining full-volume Dice/IoU reporting.

The current 200 mm contract should remain unchanged unless the project instead chooses full source-bone retention. Full retention of the longest 309 mm Ruikar scan would require approximately 396 voxels at 0.78125 mm spacing, or about 1.207 mm spacing if the 256-cubed grid is retained.

## Evidence paths

- Executed notebook: `TestProject/HPC/HPC_notebooks/modeling_HPC/00_gt_per_bone.ipynb`
- Retention CSV and summary: `TestProject/data/interim/gt_per_bone_lps_256_v1/retention_audit_v1/`
- Preprocessing crop evidence: `TestProject/data/interim/predrr_lps_256_v1/preprocessing_metadata.csv`

## Missing administrative evidence

Job ID 3715 was returned. GPU model, requested host RAM, wall time and observed peak GPU/host memory were not included and remain required with the corrected rerun bundle.
