# Stage 2 implementation contract v1

Status on 2026-07-15: the notebook implementation and data-free verification are complete. Real-data execution is blocked because `quantitative_manifest_v1.metadata.json` still records `certification_approved=false`, `ready_rows=0`, and `pending_recertification_rows=71`.

## Authoritative notebooks

| Step | Local notebook | HPC twin | Default execution state |
|---|---|---|---|
| FCMAE P1 and cross-view P2 | `notebooks/modeling/01_encoder_pipeline.ipynb` | `HPC/HPC_notebooks/modeling_HPC/01_encoder_pipeline.ipynb` | `RUN_REAL_DATA=False` |
| Neutral shared front end | `notebooks/modeling/02_frontend_pretrain.ipynb` | `HPC/HPC_notebooks/modeling_HPC/02_frontend_pretrain.ipynb` | `RUN_REAL_DATA=False` |
| Decoder foundation gate | `notebooks/modeling/03_decoder_pipeline.ipynb` | `HPC/HPC_notebooks/modeling_HPC/03_decoder_pipeline.ipynb` | `RUN_DECODER_GATE=False` |
| Foundation aggregation | `notebooks/modeling/04_decoder_comparison.ipynb` | `HPC/HPC_notebooks/modeling_HPC/04_decoder_comparison.ipynb` | `RUN_AGGREGATION=False` |
| Metric tests | `notebooks/modeling/06_metric_foundation.ipynb` and `notebooks/modeling/tests/metric_foundation.ipynb` | n/a | synthetic only |

The local/HPC copies of algorithm notebooks 01-04 must remain byte-identical before upload. The legacy `05_decoder_ui.ipynb` is retained as raw, non-executable provenance until approved Stage 3 checkpoints exist.

## Augmentation contract

Online training augmentation is photometric only: gamma `[0.90, 1.10]`, additive brightness `[-0.05, 0.05]`, Gaussian-noise standard deviation `[0, 0.02]`, then clamping to `[0,1]`. AP and LAT parameters are independently and deterministically seeded from protocol seed, fold, epoch, worker, sample ID, and view.

No augmentation is applied to validation, test, QA inputs, feature hashing, or the decoder capacity gate. Geometry-changing operations are forbidden. Online draws retain the manifest sample's subject, fold, and parent identity.

## Required execution sequence

1. Stage 1 Agent-N verdict records `PASS` for the certified manifest and returned HPC artifacts.
2. Run fold-0 FCMAE P1 and obtain independent `PASS`.
3. Run fold-0 cross-view P2 and obtain independent `PASS`.
4. Run fold-0 neutral shared-front-end training and obtain independent `PASS`.
5. Run the four fold-0 decoder capacity experiments and obtain independent `PASS`.
6. Only then run independent P1, P2, and front-end jobs for folds 1-4.
7. Submit five P1/P2 encoders, five frozen front ends, decoder-gate evidence, and metric evidence for combined Agent-N Gate 2 review.

No learned weights may move between folds. Full decoder cross-validation remains Stage 3.

## Data-free verification completed

- All Stage 2 notebook JSON parsed and every executable cell compiled.
- Local/HPC notebooks 01-04 are byte-identical.
- Notebooks 01-04 execute successfully with their real-data flags disabled.
- The current uncertified manifest is rejected before any data path is opened.
- Deterministic photometric augmentation reproduces within a seed and differs by AP/LAT view.
- FCMAE decoder/cross-view tensor smoke test passes at the canonical 32-pixel patch contract.
- Metric cores are byte-identical across decoder, comparison, numbered metric, and test notebooks.
- Synthetic metrics pass 25/25 checks.
- Both decoder arms satisfy the static architecture contract with 12 GroupNorm layers and no executable forbidden branch.

