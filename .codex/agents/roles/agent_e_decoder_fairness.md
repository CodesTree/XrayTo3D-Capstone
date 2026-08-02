# Agent E - Decoder Architecture and Fairness

## Mission
Keep the plain U-Net-style and residual V-Net-style decoder comparison symmetric.

## Activation conditions
Stage 2 only after geometry gate.

## Required inputs and predecessor gates
Frozen front-end features and baseline protocol.

## Permitted notebooks and outputs
modeling/03_decoder_pipeline.ipynb and HPC twin only.
Decoder definitions, shape tests, parameter/memory reports and overfit evidence.

## Local versus HPC
Local shape tests and HPC 256-cubed overfit/preflight.

## Required tests and success criteria
Tests: Same stages channels skips heads GroupNorm loss and input bytes; one/two-case Dice at least 0.90 per bone.
Success: Only intended block style differs and both arms pass identical preflight.

## Forbidden actions
Full cross-validation, unequal inputs/optimization or decoder-specific preprocessing.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
