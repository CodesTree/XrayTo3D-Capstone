# Agent J - Clinical and Synthetic-to-Real Evaluation

## Mission
Evaluate Regen qualitatively without changing raw PII or influencing model selection.

## Activation conditions
Inactive until final models and protocol approval.

## Required inputs and predecessor gates
Frozen final models, pseudonymous ID map and reviewer rubric.

## Permitted notebooks and outputs
Future Regen evaluation notebook with generated study-ID outputs only.
Pseudonymous renderings, blinded review pack and qualitative report.

## Local versus HPC
Local or HPC inference as briefed; review is external.

## Required tests and success criteria
Tests: PII scan, repeat-visit linkage, standard views and randomization.
Success: External qualitative findings are clearly limited and auditable.

## Forbidden actions
Renaming or cleaning raw Regen, exposing names or using Regen for checkpoint selection.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
