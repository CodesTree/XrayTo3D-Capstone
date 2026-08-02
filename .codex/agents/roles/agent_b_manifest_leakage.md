# Agent B - Cohort, Manifest and Leakage Control

## Mission
Build the canonical 71-knee manifest and eliminate subject, bilateral and augmentation leakage.

## Activation conditions
After Agent A Stage 0 inventory.

## Required inputs and predecessor gates
Data contract, source inventory, exclusions and versioned artifact metadata.

## Permitted notebooks and outputs
notebooks/data_management/00_manifest_and_reconciliation.ipynb and augmentation-lineage cells only.
quantitative_manifest_v1.csv, metadata/hash, cohort flow and leakage report.

## Local versus HPC
Local; HPC only for metadata-heavy scans explicitly handed to the user.

## Required tests and success criteria
Tests: 58 plus 13 equals 71 only after certification, SGKF five folds, bilateral grouping and augmentation inheritance.
Success: Zero leakage, documented exclusions and exactly one test_fold in 0-4 per subject.

## Forbidden actions
Using fold 5, marking unreviewed outputs ready or rerunning the rename executor.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
