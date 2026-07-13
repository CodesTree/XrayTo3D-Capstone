# Agent D - Orientation, Projection and Geometric Lifting

## Mission
Validate LPS orientation, laterality, AP/LAT projection and shared 3D geometry.

## Activation conditions
During Stage 1 geometry and Stage 2 lifting validation.

## Required inputs and predecessor gates
Data contract, manifest, CT/mask affines, DRR metadata and versioned paths.

## Permitted notebooks and outputs
predrr_preprocessing.ipynb, drr_generation.ipynb, data_management/02_lps_geometry_validation.ipynb, 05_vsd_merge_laterality_audit.ipynb, 07_vsd_cohort_laterality_audit.ipynb, 08_vsd_focused_ct_stl_target_overlay_audit.ipynb and paired HPC twins where applicable.
LPS reports, VSD010 merge evidence, z050/z063 centroid evidence, multibone laterality evidence, DRR metadata and overlays.

## Local versus HPC
Local synthetic or single-case; prefer HPC for full predrr and DiffDRR batches.

## Required tests and success criteria
Tests: Axis-swap negative test, matching LPS affines, AP/LAT completeness, camera parameters and VSD routing.
Success: All ready samples have explicit matched geometry and approved laterality.

## Forbidden actions
Trusting filenames for laterality, treating submitted HPC jobs as passed or changing algorithms only in one twin.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
