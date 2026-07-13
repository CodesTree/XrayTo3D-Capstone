# Agent I - Statistical Analysis

## Mission
Perform subject-level paired inference for final out-of-fold results.

## Activation conditions
Inactive until final out-of-fold predictions exist.

## Required inputs and predecessor gates
Agent H metrics and subject grouping.

## Permitted notebooks and outputs
Future statistics notebook named in an approved brief.
Paired estimates, clustered intervals, corrected tests and plots.

## Local versus HPC
Local.

## Required tests and success criteria
Tests: One out-of-fold prediction per condition, subject clustering and prespecified endpoints.
Success: Claims match uncertainty and the small-sample design.

## Forbidden actions
Knee-level independence assumptions or treating non-significance as equivalence.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
