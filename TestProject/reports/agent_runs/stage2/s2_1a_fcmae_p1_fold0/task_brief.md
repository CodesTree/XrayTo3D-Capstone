# Task brief - s2_1a_fcmae_p1_fold0

## Entry gate

Do not run until Agent N records Stage 1 `PASS` and the certified manifest has 71 ready rows, zero pending rows, a matching SHA-256, and zero leakage fields.

## HPC run

- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Notebook: `notebooks/01_encoder_pipeline.ipynb`
- Environment: `.venv/` created from `requirements_hpc.txt`
- First run the disposable preflight with `FOLD=0`, `RUN_REAL_DATA=True`, `RUN_PREFLIGHT=True`, `RUN_P1=False`, `RUN_P2=False`.
- After preflight review, run P1 with `RUN_PREFLIGHT=False`, `RUN_P1=True`, `RUN_P2=False`.
- Do not change the protocol hyperparameters or select an existing checkpoint.
- The staged checkpoint must match `configs/foundation_stage2_pretrained_v1.json`; runtime downloading and fallback initialization are forbidden.
- On CUDA OOM, restart from the pinned initialization with `MICRO_BATCH=8`, `ACCUM_STEPS=16`; do not resume a runtime-incompatible trainstate.

## Expected outputs

- `models/foundation_stage2_v1/fold_0/fcmae_p1_encoder.pth`
- best/last/resumable P1 trainstates
- `fcmae_p1_config.json`, `fcmae_p1_provenance.json`, and `fcmae_p1_history.csv`
- Exactly 128 uniform-with-replacement draws and one optimizer update per epoch.
- `fcmae_p1_training_curve.png`

## Pass indicators

- Strict pinned `convnextv2_tiny.fcmae` initialization with URL, license, pretrained configuration, and state SHA-256 recorded.
- Finite validation masked MSE that improves beyond the first recorded validation epoch.
- Non-collapsed validation features.
- No test path opened and no non-fold-0 learned initialization used.

