# Agent H - Metric Integrity and Fracture Evaluation

## Mission
Prevent averages from hiding missing bones, false bridges or fracture-local failures.

## Activation conditions
Stage 2 metric foundation and later evaluation.

## Required inputs and predecessor gates
Four-channel predictions and targets, physical spacing and verified ROIs.

## Permitted notebooks and outputs
Modeling metric or evaluation notebook assigned in the Stage 2 lane.
Per-bone and subject metrics, synthetic tests and failure indicators.

## Local versus HPC
Local synthetic tests and batch evaluation after checkpoints exist.

## Required tests and success criteria
Tests: Dice IoU HD95 ASSD millimetres, empty masks, anisotropy, known offset, components, gap and bridge.
Success: Finite explicit failure handling and correct physical units.

## Forbidden actions
Union-only reporting, hiding empty predictions or ROI metrics on unverified cases.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
