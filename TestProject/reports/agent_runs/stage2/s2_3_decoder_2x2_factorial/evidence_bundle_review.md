# HPC evidence-bundle review

## Verdict: PARTIAL PASS - decoder evidence complete, front-end handoff incomplete

### Decoder candidates

All eight pre-amendment candidates from folds 1 and 2 were revalidated against the downloaded
HPC results, including the actual checkpoint files. For every candidate:

- canonical `config.json` SHA-256 equals the inventory and `run_summary.json` value;
- summary, history, OOF metrics, and checkpoint file SHA-256 equal the inventory identities;
- `success=True`, seed is 42, and the frozen front end is recorded unchanged;
- 14 unique OOF knees use the expected fold and arm;
- all numeric history values are finite and each history contains 20 rows;
- GPU headroom is between 0.6436 and 0.6453, exceeding the 10% requirement.

The root `cross_validation_run_summary.csv` contains the four fold-2 rows only because the notebook
overwrites that fold-batch summary on each execution. This does not invalidate the eight
authoritative per-arm summaries.

### P2 audits

All five downloaded current P2 `feature_audit_summary.json` files match the exact current-audit
SHA-256 values in the pre-amendment inventory. They reproduce the declared dispositions:

| Fold | Structural | Quality | Local | Global | Remaining warning |
|---|---|---|---|---|---|
| 0 | PASS | WARN | PASS | PASS | `p2_preservation_warning` |
| 1 | PASS | PASS | PASS | PASS | none |
| 2 | PASS | PASS | PASS | PASS | none |
| 3 | PASS | WARN | PASS | PASS | `p2_preservation_warning` |
| 4 | PASS | WARN | PASS | WARN | `global_feature_evidence_below_threshold`, `p2_preservation_warning` |

The P2 pairing audits report PASS for all five folds. P2 provenance and histories are present, as
are the pinned-initialization/P1/P2 audit configs, summaries, local metrics, similarity matrices,
and QA figures.

### Front-end linkage

The cluster-executed inventory proves all five P2 checkpoints and shared-front-end checkpoints
matched their on-cluster provenance at inventory time. The downloaded result tree, however,
contains `shared_frontend_config.json`, history, provenance, and curve only for fold 0. The
corresponding small evidence files for folds 1-4 are not present locally.

All five recorded audit hashes differ from their current audit hashes, so all five folds require
independently reviewed `posthoc_audit_linkage_v1.json` files. No front-end retraining is indicated
by the available hash evidence.

### Remaining handoff items

Before independent fold verdicts are finalized, return for folds 1-4:

- `shared_frontend_config.json`;
- `shared_frontend_history.csv`;
- `shared_frontend_provenance.json`;
- `shared_frontend_training_curve.png`.

Also return standalone `fcmae_p2_config.json` for folds 0-4 if retained on HPC, plus explicit
decoder/P2/front-end job logs or job IDs. Existing provenance already records resource usage and
output paths, but a scheduler/notebook log remains required by the HPC result-handoff contract.
