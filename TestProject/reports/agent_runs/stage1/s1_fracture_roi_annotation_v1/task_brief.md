# Task brief: s1_fracture_roi_annotation_v1

- Stage: 1
- Assigned role: Agent C - Fracture and Target Integrity
- Independent reviewer: Agent N
- Objective: implement the manual fracture-region annotation, locked-transform replay and QA workflow for the 13 valid Ruikar knees.
- In scope: 13-case status manifest, exact recorded source CT paths, manual Slicer NIfTI ingestion, 0.5 mm LPS replay, crop/FOV retention, 256-cubed aligned ROI masks, four-bone intersection checks, visual overlays and evidence hashes.
- Out of scope: automatic fracture detection, a fifth model output, bone-target modification, AP/LAT DRR generation and Stage 2 modelling.
- Locked assumptions: no automatic ROI dilation; the annotation contains the visible gap and immediately adjacent fragments; Case3 and Case16 use the cast-cleaned source NIfTIs; verified ROIs require 100 percent retention before final nearest-neighbour resizing.
- Success criteria: exactly 13 unique status rows; every verified source ROI is binary and uses the exact source-CT grid; zero crop/FOV voxel loss; nonempty aligned output on the predrr grid; intersection with at least one target bone; user-approved QA overlay; independent Agent N review.
- Verification: compile all code cells, execute the pending-annotation baseline, pass five synthetic replay checks, confirm the notebook reports `PENDING_USER_ANNOTATION` without reading or altering raw CTs.
- Downstream: user completes manual Slicer review, then the full replay returns an HPC/local evidence bundle for Agent N.
