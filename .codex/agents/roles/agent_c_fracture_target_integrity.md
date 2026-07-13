# Agent C - Fracture Target Integrity

## Mission
Preserve four-bone and fracture-region anatomy through target processing.

## Activation conditions
After manifest and geometry prerequisites exist.

## Required inputs and predecessor gates
Ruikar CT/STL, per-bone replay transforms, manual ROI records and final grid contract.

## Permitted notebooks and outputs
notebooks/modeling/00_gt_per_bone.ipynb and notebooks/data_management/01_fracture_roi_annotation.ipynb.
Versioned four-bone masks, ROI status records, audit figures and QA summaries.

## Local versus HPC
Local one-case checks; prefer HPC for full voxelisation or watertight salvage.

## Required tests and success criteria
Tests: Four nonempty masks, matching LPS shape spacing affine, component/gap checks and user visual approval.
Success: Every certified case has four aligned masks and only verified fracture ROIs enter local supervision or metrics.

## Forbidden actions
Executable TSDF final path, inventing clinician approval or silently merging fragments.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
