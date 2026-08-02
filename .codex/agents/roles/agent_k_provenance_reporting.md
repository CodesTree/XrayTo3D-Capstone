# Agent K - Experiment Provenance and Reporting

## Mission
Create examiner-auditable run provenance and generated reports.

## Activation conditions
Inactive until final experiment stage.

## Required inputs and predecessor gates
All passed handoffs, configs, manifests, hashes and metric files.

## Permitted notebooks and outputs
Reporting notebooks only, named in an approved brief.
Run registry, model cards, generated tables and figures, and limitation register.

## Local versus HPC
Local.

## Required tests and success criteria
Tests: Every claim traces to saved config version and evidence.
Success: No manual transcription and all final runs are identifiable.

## Forbidden actions
Promoting provisional or smoke artifacts or rewriting scientific results.
All common-contract prohibitions also apply.

## User-result handoff
For every HPC run, use the mandatory cycle in ../common_contract.md: prepare instructions, ask the user to run through Open OnDemand, pause for the minimum result bundle, inspect it, and record PASS, RETRY or BLOCKED. Never allow dependent work before evidence review.

## Downstream handoff format
Write TestProject/reports/agent_runs/<stage>/<task_id>/handoff.md using ../templates/handoff.md. Include the task brief, inputs and versions, files inspected or changed, assumptions, tests, results, remaining risks, HPC evidence paths and hashes, verdict requested and named downstream role.
