# Task brief - s2_1b_crossview_p2_fold0

## Entry gate

Do not run until `s2_1a_fcmae_p1_fold0` has an independent `PASS` and its approved fold-0 P1 export and best trainstate are present with matching hashes.

## HPC run

- Working directory: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- Notebook: `notebooks/01_encoder_pipeline.ipynb`
- Configuration: `FOLD=0`, `RUN_REAL_DATA=True`, `RUN_P1=False`, `RUN_P2=True`
- The notebook must strictly load only the matching fold-0 P1 artifacts.

## Expected outputs

- `models/foundation_stage2_v1/fold_0/fcmae_p2_encoder.pth`
- best/last/resumable P2 trainstates
- `fcmae_p2_config.json`, `fcmae_p2_provenance.json`, and `fcmae_p2_history.csv`
- `fcmae_p2_training_curve.png`

## Pass indicators

- Finite and improving total validation loss.
- Non-collapsed features.
- Bidirectional paired cross-view loss is lower than bidirectional shuffled-pair loss.
- Strict fold/manifest/upstream-hash agreement and no test-path access.

