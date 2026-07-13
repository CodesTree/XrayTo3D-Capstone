# Common Contract for All Roles

## Authoritative scientific contract

- Canonical orientation is LPS end-to-end.
- Intermediate CT resampling is 0.5 mm isotropic.
- Fixed field of view is 200 mm cubed.
- Final 3D grid is 256 cubed at 0.78125 mm isotropic.
- Quantitative cohort is 58 healthy plus 13 fractured knees, total 71.
- VSD z057 Left and Right are excluded for metal artefacts.
- The target has four binary channels in this order: femur, tibia, patella, fibula.
- FCMAE P1, cross-view P2 and neutral-front-end training are fold-specific.
- The shared front end is frozen before the two decoder arms consume its features.
- Raw Regen PII is not renamed, cleaned or modified. Generated outputs use study IDs.
- Full decoder cross-validation is outside the foundation implementation.

## Work rules

1. Notebook first: pipeline work stays in documented .ipynb notebooks. Do not create pipeline .py modules.
2. Do not modify references/01-10; they are immutable reference code.
3. State assumptions and block on unresolved scientific choices.
4. Make surgical edits and preserve unrelated work.
5. A task has an explicit success criterion and verification step.
6. Read-only audits may run concurrently; overlapping notebook edits run sequentially.
7. Final configuration must reject binary-union targets, TSDF targets and fold ID 5 artifacts.
8. Do not rerun the historical destructive left/right rename executor.
9. Every runtime worker writes a handoff under TestProject/reports/agent_runs/<stage>/<task_id>/.
10. Agent N independently approves or rejects each gate; an agent cannot approve its own evidence.

## Local/HPC routing

Local work covers inventories, manifests, leakage checks, config validation, synthetic geometry/metric tests and safe CPU smoke tests. Prefer HPC for full merge/preprocessing/voxelisation/DRR/augmentation batches, FCMAE, front-end training and 256-cubed decoder preflight.

HPC notebooks use Linux paths rooted at:

/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2 _24020059/

Local twins are under TestProject/HPC/HPC_notebooks/. Only cells tagged environment-configuration may differ from their local pair; algorithm cells must hash identically.

## Mandatory HPC handoff

Before a run, give the user the notebook path, cells/config to verify, environment, working directory, expected outputs and pass/failure indicators. Ask the user to run it through Open OnDemand, then pause.

The returned bundle must contain the executed notebook or log, run config JSON, summary CSV/JSON, QA figures/overlays, traceback if any, GPU model, requested host RAM and wall time, observed peak GPU/host memory where available, job ID, bulk-output path, and hashes for large outputs.

Never infer success from submission or file existence. Review the bundle and issue exactly one verdict: PASS, RETRY or BLOCKED. Record evidence and verdict before dependent HPC work starts.
