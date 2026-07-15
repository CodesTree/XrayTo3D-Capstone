# Task brief - s2_1a_fcmae_p1_fold0

## Entry gate

Do not run until Agent N records Stage 1 `PASS` and the certified manifest has 71 ready rows, zero pending rows, a matching SHA-256, and zero leakage fields.

## HPC run

- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Notebook: `notebooks/01_encoder_pipeline.ipynb`
- Environment: `.venv/` created from `requirements_hpc.txt`
- Configuration: `FOLD=0`, `RUN_REAL_DATA=True`, `RUN_P1=True`, `RUN_P2=False`
- Do not change the protocol hyperparameters or select an existing checkpoint.
- On CUDA OOM, halve `MICRO_BATCH`, double `ACCUM_STEPS`, preserve effective batch 128, record the change, and resume from the matching last trainstate.

## Expected outputs

- `models/foundation_stage2_v1/fold_0/fcmae_p1_encoder.pth`
- best/last/resumable P1 trainstates
- `fcmae_p1_config.json`, `fcmae_p1_provenance.json`, and `fcmae_p1_history.csv`
- `fcmae_p1_training_curve.png`

## Pass indicators

- Strict pinned `convnextv2_tiny.fcmae` initialization with URL, license, pretrained configuration, and state SHA-256 recorded.
- Finite validation masked MSE that improves beyond the first recorded validation epoch.
- Non-collapsed validation features.
- No test path opened and no non-fold-0 learned initialization used.

