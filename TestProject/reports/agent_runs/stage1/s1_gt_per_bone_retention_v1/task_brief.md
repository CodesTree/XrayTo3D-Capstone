# Task brief: s1_gt_per_bone_retention_v1

- Stage: 1
- Assigned role: Agent C - Fracture and Target Integrity
- Independent reviewer: Agent N
- Change class: methodological correction
- Objective: certify preservation of femur, tibia, patella and fibula anatomy through the recorded crop, fixed-FOV, resize and orientation transform.
- In scope: mask-based retention, removed-component analysis, saved-mask replay, predrr-grid checks and focused visual evidence for the eight previously warned fractured cases.
- Out of scope: fracture-ROI creation, AP/LAT DRR generation, manifest edits, Stage 2 modelling and any modification of the approved masks.
- Predecessor gates and evidence: corrected `predrr_lps_256_v1` has passed its targeted geometry retry; 71 samples and 284 saved per-bone masks are available.
- Inputs and versions: `predrr_lps_256_v1`, `gt_per_bone_lps_256_v1`, four-bone order `[femur, tibia, patella, fibula]`, LPS, 0.5 mm source grid, 200 mm FOV and 256 cubed final grid.
- Permitted notebooks/outputs: local and HPC `00_gt_per_bone.ipynb` twins; v1 evidence remains under `retention_audit_v1/`; corrected evidence is written only under `data/interim/gt_per_bone_lps_256_v1/retention_audit_v2/`.
- Local or HPC: syntax, parity and synthetic boundary tests locally; full 71-sample audit on HPC.
- Assumption requiring confirmation: retention is measured from the STL rasterized inside its version-matched 0.5 mm CT grid. The earlier CT-STL overlay audit is the evidence that the CT grid covers the intended bone anatomy.
- Success criteria: 71 samples and 284 rows; ROI-crop retention at least 0.995 with no removed 6-connected component of 30 or more voxels; large FOV loss allowed only for boundary-connected femur/tibia/fibula shafts clipped solely along S-I; exact replay, binary/non-empty and predrr-grid checks; user approval of all fractured-case overlays; fracture-ROI certification remains a separate 100 percent gate.
- Verification: compile every code cell, confirm local/HPC algorithm-source identity, run synthetic 29/30-voxel and connected/disconnected cut-plane tests, then inspect the returned v2 HPC CSV/JSON/hash manifest and overlays.
- Downstream role: Agent N for the independent retention-gate verdict, then Agent C for fracture-region masks.
- Overlap check: this lane edits only the two `00_gt_per_bone.ipynb` twins and this task report; no Stage 2 notebook is touched.

