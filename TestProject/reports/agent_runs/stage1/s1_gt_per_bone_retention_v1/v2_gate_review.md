# Per-bone retention v2 gate review

- Producing role: Agent C - Fracture and Target Integrity
- Run date: 2026-07-14 UTC
- HPC job ID: not supplied
- Automated bone-retention verdict: PASS
- Full Stage 1 target verdict: PENDING fracture-ROI masks and user visual approval
- Agent N Gate 1 review: not yet eligible

## Evidence integrity

- The HPC notebook executed `retention-run-v2` at execution count 11 with no error output.
- Local and HPC algorithm sources are byte-identical at SHA-256 `5ee1f0b3d0da567148cf127997596139a221f1671a9ce4521c106ae2c027f42f`.
- All 16 returned v2 evidence hashes match.
- The v2 table contains 71 unique samples, 284 unique sample/bone rows, 58 healthy knees, 13 fractured knees and exactly 71 rows for each bone.
- The local notebook copy has the same source but its v2 configuration and run cells have no execution count or outputs. It is not evidence of a completed local v2 run; this does not invalidate the completed HPC result.

## Automated gates

- ROI-crop failures: 0.
- Unexpected FOV failures: 0.
- Replay-to-saved mismatches: 0.
- Binary/non-empty failures: 0.
- Predrr-grid failures: 0.
- Automated bone failures: 0.
- Automated bone verdict reported by notebook: `PASS`.

## False-positive investigation

The 39 fractured long-bone rows are not failure rows. They are correctly classified as `expected_boundary_connected_long_bone_shaft`:

- 13 femora connect across the superior FOV cut plane.
- 13 tibiae and 13 fibulae connect across the inferior FOV cut plane.
- Every row has exactly one significant removed component.
- Every significant component touches the retained shaft across the cut plane.
- Minimum measured cut-plane contact is 176 voxels.
- All are clipped only along S-I; none are clipped along x or y.

The two small non-zero exceptions are within the locked tolerance and are not anatomical failures:

- `Case3_PartLeft/fibula`: one 0.5 mm ROI-crop voxel removed; ROI retention 0.9999907.
- `Case13_PartRight/patella`: one 0.5 mm FOV voxel removed; FOV retention 0.9999909.

Visual review of the 13 returned overlays shows orange exclusion confined to remote long-bone shaft ends. No material red internal ROI loss or removed patellar/fracture-region anatomy is visible. The one-voxel exceptions are below visible anatomical significance.

The Job 3715 v1 failures were therefore false-positive failures caused by combining deliberate 200 mm FOV exclusion with unintended crop loss. Version 2 now reports those exclusions accurately without hiding them.

## Scope and modelling consequence

The four-channel targets are certified for automated bone integrity inside the locked 200 mm knee reconstruction domain. This does not certify complete source long bones and does not yet certify fracture preservation. Boundary-censored surface metrics remain required for HD95/ASSD so artificial shaft cut planes do not dominate evaluation.

## Required next step

Proceed to manual fracture-region masks for the 13 valid Ruikar cases:

1. draw each ROI in original CT LPS space;
2. label each case `verified_fracture`, `no_visible_fracture` or `ungradable`;
3. replay the ROI through the identical crop, FOV, resize and orientation transform;
4. require 100 percent retention for every verified ROI;
5. obtain user visual approval before fracture-local supervision or metrics;
6. then request the independent Agent N target/geometry gate review.

The AP/LAT DRR dependency and full Stage 1 gate remain pending.

## Administrative gaps

The v2 HPC job ID, GPU model, requested host RAM, wall time and observed peak memory were not supplied. Record these when available; they do not change the scientific automated bone-gate verdict.
