# Task brief - s2_3_decoder_preflight

## Entry gate

Do not run until the fold-0 frozen front end and its clean two-case feature cache have an independent `PASS`.

## HPC run

- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Notebook: `notebooks/03_decoder_pipeline.ipynb`
- Configuration: `RUN_DECODER_GATE=True`
- Run all four arm/subset combinations in the notebook without reducing 256-cubed resolution.

## Expected outputs

- `models/decoders/foundation_stage2_v1/fold_0/foundation_summary.json` and `.csv`
- per-run `config.json`, `history.csv`, `capacity_curve.png`, update-5 resume checkpoint, and final checkpoint
- parameter, pre-update-logit, config, feature-cache, and checkpoint hashes
- wall time, peak GPU memory, peak host memory, total GPU memory, and headroom

## Pass indicators

- Every case and every bone reaches hard Dice at least 0.90 at two consecutive evaluations in all four runs.
- Save/reload into fresh model/optimizer/scheduler/scaler preserves parameter and pre-update-logit hashes and training continues.
- All rows use one identical feature-cache SHA-256.
- GPU-memory headroom is at least 10 percent.

