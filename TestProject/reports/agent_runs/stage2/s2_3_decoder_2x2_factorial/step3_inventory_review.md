# Step 3 pre-amendment inventory review

## Verdict: PASS

Reviewed inputs:

- executed `03b_decoder_cross_validation.ipynb`, SHA-256
  `95657312ff496c8085f76218a67d82c5b6ade6da22e9b883ee36a9a87d1b84d8`;
- `pre_amendment_decoder_run_inventory.json`, SHA-256
  `dbe1b0eca8195df9a26d057063c26eb211c54420e1f527437a9fbb38137aa907`;
- amendment SHA-256
  `ac9edb4774d5b441814b49037804624343046d9c9ca2aea0282954754f7449b4`.

The executed inventory notebook used CUDA, ran with `RUN_EXISTING_RUN_AUDIT=True` and decoder
training disabled, emitted the supplied inventory, and recorded no error output.

## Completed reuse candidates

Eight exact-identity candidates were found:

- fold 1: all four arms;
- fold 2: all four arms.

Every candidate records seed 42 and a fold-consistent shared-front-end SHA-256. These candidates
are eligible for independent acceptance, not automatically accepted. The combined Stage 2 verdict
must reproduce each accepted identity exactly. The four fold-1 arms share front end
`d6509c1698db6c02e78d6253c52f8e443eea95523a57d0226923df6577698e34`; the four fold-2 arms
share front end `9c85e9fb8085a67d5ad1e2ba615c0077800b321ca8b8a6033de2ea52ab744370`.

No complete run was found for folds 0, 3, or 4. Their twelve entries remain scheduled for final
training after the combined gate. Before training, inspect the HPC directories for partial files;
the inventory classifies absent and incomplete states together.

## Front-end disposition

For all five folds:

- the P2 checkpoint SHA-256 matches shared-front-end provenance;
- the shared-front-end checkpoint SHA-256 matches provenance;
- the current P2 audit SHA-256 differs from the audit recorded at front-end export.

Therefore no front-end retraining is required on checkpoint-integrity grounds. All five folds do
require independently reviewed `posthoc_audit_linkage_v1.json` files binding the recorded audit,
current audit, and current front-end checkpoint hashes. Historical provenance must not be edited.

## Evidence still required for independent acceptance

The inventory proves that matching files existed and passed its structural/hash checks on HPC, but
does not expose their contents locally. Before the combined verdict accepts any fold-1/fold-2 run,
return the following for all eight candidates:

- `config.json`;
- `run_summary.json`;
- `history.csv`;
- `oof_metrics.csv`;
- training/job log and job/output path;
- resource-usage record;
- checkpoint file size and SHA-256 (the large checkpoint itself need not be transferred).

For fold-specific QA and audit linkage, also return each fold's current P2 audit summary, shared
front-end provenance, relevant QA figures, P2/front-end configuration and histories, resource
usage, job/output paths, and hashes.
