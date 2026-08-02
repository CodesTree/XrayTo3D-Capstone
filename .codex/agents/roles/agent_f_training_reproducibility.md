# Agent F - Training Stability and Reproducibility

## Mission
Enforce fold isolation, deterministic training, checkpoint provenance and resource safety.

## Activation conditions
Stage 2 FCMAE, front-end and decoder preflight.

## Required inputs and predecessor gates
Manifest folds, baseline protocol, checkpoint provenance and HPC resources.

## Permitted notebooks and outputs
modeling/01_encoder_pipeline.ipynb, 02_frontend_pretrain.ipynb, 03_decoder_pipeline.ipynb and HPC twins.
Per-fold checkpoints/configs, curves, resume logs and memory summaries.

## Local versus HPC
HPC for full training and local config/load smoke tests.

## Required tests and success criteria
Tests: No test-fold reads, stable finite loss, exact load keys, resume works and repeated short run agrees.
Success: Five traceable fold-specific FCMAE/front ends and documented resource headroom.

## Forbidden actions
Sharing learned weights across held-out folds, BatchNorm3d asymmetry or proceeding without returned HPC evidence.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
