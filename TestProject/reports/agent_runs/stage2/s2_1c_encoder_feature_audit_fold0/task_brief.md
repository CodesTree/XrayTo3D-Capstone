# Task brief - s2_1c_encoder_feature_audit_fold0

## Entry gate

Run the pinned-initialization reference after the official file is staged. Run P1/P2 audits only after the corresponding fold-0 training stage has an independent PASS. The certified manifest and matching fold export must be present.

## HPC run

- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Notebook: `notebooks/01b_encoder_feature_audit.ipynb`
- P1 configuration: `AUDIT_FOLD=0`, `AUDIT_STAGE="P1"`, `RUN_AUDIT=True`
- Reference configuration: `AUDIT_FOLD=0`, `AUDIT_STAGE="pinned_init"`, `RUN_AUDIT=True`
- P2 configuration: `AUDIT_FOLD=0`, `AUDIT_STAGE="P2"`, `RUN_AUDIT=True`
- Do not expose test-fold paths. The encoder must remain frozen and hash-identical.
- Targets are evaluation-only and may only create the four-bone AP/LAT silhouettes.

## Required evidence

Return the executed notebook/log, config, feature manifest, local probe metrics/history, retrieval matrix, feature-audit summary, QA PNGs, resource usage, HPC paths, sizes and SHA-256 records.

## Interpretation

- Structural status must be PASS: provenance, finite/non-collapsed features, frozen encoder and unopened test paths.
- Local/global quality may be PASS or WARN; WARN is retained for scientific interpretation and does not automatically block P2/front-end work.
- P2 preservation records a WARN when local macro Dice drops by more than 0.02 or global MRR does not improve.
