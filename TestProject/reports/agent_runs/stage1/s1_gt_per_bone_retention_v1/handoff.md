# Handoff: s1_gt_per_bone_retention_v1

- Stage: 1
- Producing role: Agent C - Fracture and Target Integrity
- Downstream role: Agent N
- Current status: PASS_AUTOMATED_BONE_GATE - PENDING_FRACTURE_ROI_AND_USER_REVIEW
- Verdict requested after returned evidence: PASS | RETRY | BLOCKED

## Problem statement

The preprocessing notebook's former `<0.95` warning measures high-HU foreground and support removal, not preservation of the four target bones. The target gate therefore needs a direct STL-mask audit before the target set can be certified.

## Files changed

- `TestProject/notebooks/modeling/00_gt_per_bone.ipynb`
- `TestProject/HPC/HPC_notebooks/modeling_HPC/00_gt_per_bone.ipynb`

## Implemented checks

- Rasterize each source STL on its version-matched oriented 0.5 mm CT grid.
- Calculate the exact source voxels surviving the recorded ROI crop and centered 200 mm FOV.
- Require at least 99.5 percent retention for every sample/bone row.
- Label removed voxels with 6-connectivity and reject any removed component of at least 30 voxels.
- Replay crop, fixed FOV, nearest-neighbour resize and S-I flip; require byte-identical equality with the saved mask.
- Require each saved mask to be binary and non-empty and to share the predrr size, spacing, origin and direction.
- Render AP/LAT overlays for the eight previously warned fractured cases and every automatic failure. Green is the complete pre-FOV STL mask; red is removed anatomy.
- Write CSV, summary JSON, run-configuration JSON, QA PNGs and a SHA-256 evidence manifest without overwriting target masks.

## Job 3715 v2 correction

- Writes only to `retention_audit_v2`; the v2 code never targets the v1 folder.
- Separates strict ROI-crop retention from descriptive/policy-controlled fixed-FOV truncation.
- Large FOV exclusion passes only for femur, tibia or fibula, only for S-I-only clipping, and only when every significant removed component is 6-connected to the retained shaft across a cut plane.
- Small FOV loss remains subject to the original 99.5 percent and 30-voxel tolerance.
- QA uses red for ROI-crop loss and orange for fixed-FOV exclusion.
- Reports `automated_bone_verdict` separately from `stage1_target_verdict`; the latter remains pending until manual fracture ROIs retain 100 percent and the user approves QA.

## Local verification

- Both notebooks parse as JSON and all 21 code cells compile.
- All algorithm cell sources are identical between the local and HPC twins.
- Code-source SHA-256: `5ee1f0b3d0da567148cf127997596139a221f1671a9ce4521c106ae2c027f42f`.
- Synthetic ROI/FOV slice accounting passed.
- Synthetic connected shaft components at both S-I cut planes are accepted.
- A disconnected 30-voxel component outside the FOV is rejected.
- The original 29/30-voxel tolerance boundary remains enforced.
- `git diff --check` passed for the notebook edits.
- A complete local one-case replay was not run because the workstation environment lacks `trimesh` and its Matplotlib binary is incompatible with installed NumPy. Computational geometry-policy helpers were tested independently; the full real-data run is routed to HPC.

## Success-criterion assessment

- Implementation and local static/synthetic checks: PASS.
- Job 3715 evidence completeness and integrity: PASS for 71 samples, 284 rows and 16 returned evidence hashes.
- Full-source-to-FOV retention gate: RETRY because it treats the deliberate 200 mm field-of-view boundary as accidental anatomy loss.
- Corrected v2 automated bone gate: PASS; 71 samples, 284 rows and zero automated failures.
- Fracture-anatomy visual and ROI gates: PENDING USER APPROVAL AND MANUAL ROI CREATION.
- Agent N independent verdict: PENDING RETURNED BUNDLE.

## Concurrent v1 evidence note

After the Job 3715 review, a separate v1 run started at `2026-07-14T07:58:14Z` and rewrote `run_configuration.json` plus the 13 fractured QA PNGs without completing a new CSV, summary or hash manifest. The original returned bundle was hash-clean when reviewed, but the current local v1 folder is now a mixed partial rerun and must not be used as the v2 result bundle. The v2 notebook writes to a separate directory.

## Remaining risks and prohibited next steps

- Rasterization counts anatomy within the version-matched CT grid; intended STL coverage relies on the already completed CT-STL overlay audit.
- Do not infer success from notebook completion or output existence.
- Do not certify the targets, start fracture-local metrics, generate dependent DRRs or begin Stage 2 from this implementation-only handoff.

## Producer conclusion

The returned v2 HPC evidence passes the automated bone-retention subgate. Proceed to manual fracture-region masks; full Stage 1 and Agent N Gate 1 remain pending fracture-ROI retention, user approval and certified AP/LAT DRRs.

