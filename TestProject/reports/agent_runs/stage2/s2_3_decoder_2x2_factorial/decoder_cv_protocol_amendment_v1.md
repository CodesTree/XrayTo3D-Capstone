# Decoder cross-validation protocol amendment v1

Status: frozen before any fold-0, fold-3, or fold-4 decoder result is inspected.

This amendment supplements, but does not rewrite, `implementation_contract_v1.md`. It resolves the
Stage 2 decoder-capacity versus full-cross-validation notebook-role conflict and pre-registers the
handling of the fold-specific P2 feature-audit warnings. Historical audit summaries and
`quality_status` values remain unchanged.

## Scope and observed information

- The decoder comparison is a matched 2x2 factorial of activation (`ReLU`, `PReLU`) and residual
  topology (plain, residual), conditional on the fold-specific frozen front ends.
- Before this amendment, all four fold-2 decoder arms were completed and at least one fold-1 arm
  was completed. These results are treated as pre-amendment runs. They may enter the final analysis
  only after exact artifact/configuration hashes are inventoried and explicitly accepted by the
  independent combined Stage 2 verdict. No architecture, loss, optimiser, seed, split, endpoint, or
  decision rule may be changed in response to those observed results.
- No fold-0, fold-3, or fold-4 decoder result informed this amendment.

## Fold dispositions requested from independent review

- Fold 0: accept for decoder comparison if structural audit checks remain PASS. The remaining
  preservation warning is a threshold-edge MRR fluctuation and must remain visible in provenance.
- Fold 3: accept for decoder comparison if structural audit checks remain PASS. Record both the
  threshold-edge preservation warning and the L2 effective-rank caveat.
- Fold 4: conditional acceptance only. Record the failed global feature evidence, P2 preservation
  warning, and L2/L3 dimensional-collapse evidence. Acceptance is scoped to a matched decoder
  comparison and does not establish adequate absolute front-end quality.
- Folds 1 and 2: independent verdicts are still required for the combined gate, even when their
  current automated quality status is PASS.

## Authoritative notebook roles

Local and HPC twins must be byte-identical before upload.

| Phase | Authoritative notebook |
|---|---|
| Four-arm capacity and save/resume gate | `03a_decoder_capacity_gate.ipynb` |
| Controlled five-fold decoder training | `03b_decoder_cross_validation.ipynb` |
| OOF aggregation and sensitivity analysis | `04_decoder_comparison.ipynb` |

The earlier `03_decoder_pipeline.ipynb` copies remain non-authoritative provenance. Full decoder
cross-validation may start only after all eight capacity runs pass and an independent combined
Stage 2 verdict binds the current audit, fold-verdict, capacity-summary, and amendment hashes.

## Primary analysis

- Cohort: all 71 knees and 43 subjects across all five test folds.
- Arms: `plain_unet_style`, `residual_vnet_style`, `residual_relu_style`, and
  `plain_prelu_style`.
- Seed: 42.
- Co-primary endpoints: subject-level macro Dice and macro ASSD.
- Co-primary contrasts: residual main effect and activation main effect, with Holm correction over
  the four contrast-endpoint tests.
- A factor is preferred only when both corrected endpoints agree in direction and are significant,
  and the residual-by-activation interaction is not significant. Otherwise it is `INCONCLUSIVE`.
- The diagonal V-minus-U contrast is retained for continuity but is secondary because it combines
  two factors.

## Mandatory fold-4-excluded sensitivity analysis

- Filter by `test_fold != 4`; do not select subjects by decoder outcome.
- Derive counts from the OOF table. Under the certified current split the expected counts are 57
  knees and 34 subjects.
- Re-run the predefined contrasts and Holm family as sensitivity evidence.
- The sensitivity analysis cannot independently select a preferred factor or decoder.
- Matching primary and sensitivity conclusions are labelled `ROBUST`.
- A primary preference that is not reproduced is labelled `FOLD4_DEPENDENT`.
- Opposite directions, or a preference appearing only after fold-4 exclusion, produce an overall
  `INCONCLUSIVE` interpretation.

## Existing-run acceptance

The clean CV notebook emits a hash inventory of completed pre-amendment runs. A run is reusable only
when the independent combined verdict includes an exact identity containing its fold, arm,
configuration SHA-256, run-summary SHA-256, checkpoint SHA-256, OOF-metrics SHA-256, shared-front-end
SHA-256, and seed. Missing, mismatched, incomplete, or unaccepted runs are pilot-only and must be
rerun. Interrupted run directories must be archived outside the authoritative output tree before a
new run starts; they must not be silently resumed under the amended protocol.

## Claims boundary

The final inference compares decoder factors under fixed fold-specific representations. Weakness in
a shared front end cannot give the four arms different inputs within a fold, but it can change task
difficulty, absolute performance, and decoder-by-representation interaction. Results therefore do
not establish a universally superior complete U-Net/V-Net system or clinical utility.

