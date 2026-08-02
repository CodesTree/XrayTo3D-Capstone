# Decision record: stage1-drr-resource-evidence-waiver

- Date: 2026-07-16
- Owner: Project Orchestrator
- Status: accepted
- Change class: evidence-policy correction
- Context: The Stage 1 DRR scientific gate is determined by manifest-aligned inventory, projection configuration, numerical integrity, artifact hashes, QA figures and independent review. Job ID, GPU model, requested RAM and wall time describe execution logistics but do not change the validity of the saved DRRs.
- Decision: Job ID, GPU model, requested RAM and wall time remain best-effort provenance fields but are not required for Stage 1 DRR PASS. Their absence must be recorded and must not change `PASS_READY_FOR_AGENT_N` when all scientific and artifact-integrity checks pass.
- Alternatives considered: retain the fields as blocking; remove resource capture entirely. Both were rejected because the first blocks a scientifically complete historical run and the second discards useful provenance.
- Scientific impact: none. DRR geometry, cohort coverage, numerical QA, hashes, configuration, visual QA and Agent N review remain mandatory.
- Files/configs affected: local/HPC DRR evidence cells and `s1_drr_generation_evidence_v1` task records.
- Required symmetric changes: identical evidence logic in local and HPC notebook twins.
- Evidence: 71 manifest cases, 142 NPY/PNG pairs, 142 metadata rows, zero local numerical failures and 147 recomputed evidence hashes.
- Agent N review: required before Stage 1 certification.

This waiver is limited to the Stage 1 DRR evidence gate. It does not relax Stage 2 training, capacity, memory-headroom, save/resume or reproducibility evidence requirements.