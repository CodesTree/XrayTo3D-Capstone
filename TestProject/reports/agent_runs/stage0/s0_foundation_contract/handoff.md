# Handoff: s0_foundation_contract

- Stage: 0
- Producing roles: Orchestrator and Agent A
- Downstream role: Agent N
- Verdict requested: PASS

## Files and evidence
- .codex/agents registry, common contract, 14 role files and five templates
- TestProject/configs/data_contract_v1.json
- TestProject/configs/baseline_protocol_v1.json
- TestProject/reports/repository_audit/artifact_inventory_v1.csv
- TestProject/reports/repository_audit/legacy_quarantine_v1.csv
- TestProject/reports/repository_audit/notebook_parity_v1.json
- TestProject/notebooks/data_management/04_contract_validation.ipynb
- TestProject/notebooks/data_management/06_artifact_inventory.ipynb

## Tests observed
- Agent registry: 14 expected IDs, Agent M absent, G/I/J/K inactive.
- Static contract validation: PASS.
- Preprocessing local/HPC algorithm parity: PASS for predrr and DiffDRR pairs.
- Artifact inventory: 382 classified rows; 34 legacy/smoke items quarantined by configuration.
- No raw data filenames were inventoried; no raw Regen files were changed.

## Remaining risks
Stage 1 data, laterality, ROI and HPC output certification is separate and still pending. Stage 2 belongs to the Claude Code lane.
