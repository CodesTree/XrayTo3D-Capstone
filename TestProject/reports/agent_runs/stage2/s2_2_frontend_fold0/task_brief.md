# Task brief - s2_2_frontend_fold0

## Entry gate

Do not run until the matching fold-0 P2 encoder has an independent `PASS` and the P2 feature audit has structural status `PASS`. Audit quality warnings are recorded but do not automatically block this phase.

## HPC run

- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Notebook: `notebooks/02_frontend_pretrain.ipynb`
- Configuration: `FOLD=0`, `RUN_REAL_DATA=True`
- Train encoder, fusion, lift, and neutral head together for 40 epochs; do not freeze the encoder during this phase.

## Expected outputs

- head-free `models/foundation_stage2_v1/fold_0/shared_frontend.pth`
- best/last/five-epoch trainstates, config, provenance, history, and training curve
- clean deterministic `frozen_feature_cache.pt` and cache manifest
- per-level feature hashes for `VSD_016_Left` and `Case14_PartRight`
- clean `frozen_feature_qa.png`

## Pass indicators

- Selection by highest validation macro per-bone Dice with lower loss as the tie-breaker.
- Strict head-free reload, all parameters frozen, evaluation mode, and identical repeated clean-feature hashes.
- Exact `[64,128,256,512]` feature channels at `[64-cubed,32-cubed,16-cubed,8-cubed]`.

