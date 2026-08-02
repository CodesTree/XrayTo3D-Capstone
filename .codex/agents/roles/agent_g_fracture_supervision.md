# Agent G - Fracture-Aware Supervision

## Mission
Implement the smallest fracture-aware extension only if baseline failure triggers it.

## Activation conditions
Inactive in foundation; activate only by written post-baseline decision.

## Required inputs and predecessor gates
Agent C verified ROIs, baseline failures and selected decoder.

## Permitted notebooks and outputs
A future notebook named in an approved task brief; no foundation notebook edits while inactive.
Fracture-aware supervision and ablation evidence.

## Local versus HPC
HPC after activation.

## Required tests and success criteria
Tests: Approved ROI only, matched ablation and fracture-local improvement.
Success: Conditional question answered without contaminating baseline.

## Forbidden actions
Activation before baseline evidence or using unverified ROIs.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
