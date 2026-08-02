# Agent A - Repository Audit and Codebase Cleanup

## Mission
Inventory and classify the repository while preserving notebook-first development.

## Activation conditions
Stage 0 or when new artifact classes appear.

## Required inputs and predecessor gates
Repository tree, configs, notebook metadata and historical reports.

## Permitted notebooks and outputs
May edit data_management/03_notebook_environment_parity.ipynb, 04_contract_validation.ipynb and 06_artifact_inventory.ipynb; other notebooks are read-only.
reports/repository_audit inventory, duplicate map and legacy quarantine report.

## Local versus HPC
Local and read-heavy.

## Required tests and success criteria
Tests: Every artifact classified, one authoritative item per function and local/HPC parity gate.
Success: No unclassified artifact and no final config can select fold 5, binary union or TSDF.

## Forbidden actions
Deleting artifacts, converting pipeline notebooks to modules or modifying references/01-10.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
