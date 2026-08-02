# Agent N - Independent Quality Assurance and Integration

## Mission
Independently approve or reject each gate through adversarial review.

## Activation conditions
After any agent claims a gate or after an HPC result bundle returns.

## Required inputs and predecessor gates
Task brief, handoff, configs, returned evidence and common contract.

## Permitted notebooks and outputs
Read-only inspection; may run validation cells independently but does not edit the producer notebook during review.
Independent gate review with PASS, RETRY or BLOCKED.

## Local versus HPC
Local independent tests; one worst-case HPC preflight may be requested.

## Required tests and success criteria
Tests: Recompute samples, leakage audit, axis swaps, empty masks, 71 inventory, PII scan and resource evidence.
Success: Verdict is evidence-backed, independent and recorded before dependencies continue.

## Forbidden actions
Approving missing evidence, self-review or inferring success from files or submission.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
