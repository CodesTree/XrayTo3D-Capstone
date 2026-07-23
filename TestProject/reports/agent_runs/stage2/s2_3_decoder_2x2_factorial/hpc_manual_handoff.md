# HPC manual handoff - decoder comparison

## Preserve current HPC state

1. Download the currently executed `notebooks/03_decoder_pipeline.ipynb` and its log as pilot
   provenance if not already retained locally.
2. Do not delete completed fold-2 or fold-1 decoder artifacts.
3. Do not overwrite or edit historical feature-audit summaries, front-end provenance, or
   `quality_status` values.

## Upload

Copy these files to the HPC project while preserving the relative destination shown in
`upload_manifest.sha256`:

- `notebooks/03a_decoder_capacity_gate.ipynb`
- `notebooks/03b_decoder_cross_validation.ipynb`
- `notebooks/04_decoder_comparison.ipynb`
- `reports/agent_runs/stage2/s2_3_decoder_2x2_factorial/decoder_cv_protocol_amendment_v1.md`
- the three `*.template.json` files in the same report directory

Verify all four fixed upload hashes before execution. The JSON templates are instructions, not
PASS verdicts.

## Inventory existing work first

In `03b_decoder_cross_validation.ipynb`, set only:

```python
RUN_EXISTING_RUN_AUDIT = True
RUN_CROSS_VALIDATION = False
RUN_SEED_SWEEP = False
```

Execute the notebook. Return
`reports/agent_runs/stage2/s2_3_decoder_2x2_factorial/pre_amendment_decoder_run_inventory.json`,
the executed notebook, and log. The inventory identifies every complete fold-2/fold-1 candidate
by exact hashes and reports whether each front end needs post-hoc audit linkage or has a P2/front-end
hash mismatch.

If `p2_checkpoint_matches_provenance` or `frontend_checkpoint_matches_provenance` is false, stop for
that fold. The matching Stage 2 front end must be regenerated/reviewed before decoder use. If only
`requires_posthoc_audit_linkage` is true, request an independent
`posthoc_audit_linkage_v1.json`; do not rewrite the original provenance.

## Obtain fold-specific independent verdicts

Submit the current audits, QA figures, training histories, pairing evidence, front-end provenance,
hash inventory, resource records, job paths, and amendment hash for independent review. Place the
returned verdict for every fold at:

`models/foundation_stage2_v1/fold_{fold}/feature_audit/independent_qa_verdict.json`

Fold 4 must include all three required sensitivity/claims-boundary conditions. These verdicts are
required even for automatically passing folds 1 and 2.

## Run the eight-result capacity gate

In `03a_decoder_capacity_gate.ipynb`, set `RUN_DECODER_GATE=True` and execute on the HPC GPU. It must
produce four arms x two subsets, with all eight results passing capacity, resume, common-cache, and
memory-headroom checks.

Return the executed notebook/log, `foundation_summary.json` and CSV, eight configs/histories/curves,
resume/final checkpoint hashes, feature-cache hash, resource usage, and job/output paths.

## Obtain combined independent Stage 2 PASS

Submit the capacity bundle, all five fold verdicts, amendment hash, linkage files, and existing-run
inventory. The reviewer decides which complete pre-amendment identities are accepted. Place the
returned file at:

`models/foundation_stage2_v1/independent_stage2_gate_verdict.json`

Do not copy candidate identities into this verdict without independent acceptance.

## Archive incomplete pilot directories

Before final training, compare the inventory's `absent_or_incomplete` entries with the decoder
output tree. Move any partially populated pre-amendment arm directory outside
`models/decoders/foundation_stage2_v1/fold_{fold}/{arm}` into a clearly labelled pilot archive.
Never archive a completed candidate that the combined verdict accepts. This prevents incompatible
old resume checkpoints from being loaded under the amended QA configuration.

## Run remaining folds

In `03b_decoder_cross_validation.ipynb`, reset `RUN_EXISTING_RUN_AUDIT=False`. Run one fold per HPC
execution with `RUN_CROSS_VALIDATION=True`, `RUN_SEED_SWEEP=False`, and one of:

```python
FOLDS_TO_RUN = [0]
FOLDS_TO_RUN = [1]
FOLDS_TO_RUN = [2]
FOLDS_TO_RUN = [3]
FOLDS_TO_RUN = [4]
```

The notebook will skip only exact identities accepted by the combined verdict. Therefore the four
completed fold-2 arms should not retrain if independently accepted. Any missing or unaccepted arm
must run normally. After each fold return the executed notebook/log, four configs, summaries,
histories, OOF metric files, checkpoint hashes, QA hashes, resources, and job/output paths.

## Aggregate

After all twenty arms are complete or independently accepted, execute
`04_decoder_comparison.ipynb` with `RUN_AGGREGATION=True`. It will independently revalidate the
combined gate and accepted old-run identities, then produce:

- primary all-five-fold results (71 knees, 43 subjects);
- fold-4-excluded sensitivity results (57 knees, 34 subjects);
- a robustness interpretation that prevents sensitivity-only winner selection;
- co-primary, per-bone, cohort, topology, pathology, resource, and optional variance outputs.

Return the executed notebook/log and the complete
`reports/decoder_comparison/foundation_stage2_v1/` directory. The optional seed sweep follows only
after the main comparison is secured.
