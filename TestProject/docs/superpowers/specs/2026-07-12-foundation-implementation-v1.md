# Foundation Implementation Specification v1

**Date:** 2026-07-12  
**Status:** Authoritative for Stages 0–2  
**Supersedes:** the data, orientation, encoder-split and HPC-handoff sections of the external v5 cleanup specification.

## Scientific contract

The foundation prepares a controlled comparison of a plain U-Net-style decoder and a residual V-Net-style decoder. Both conditions use the same fold-specific ConvNeXtV2 encoder, FCMAE P1 and cross-view P2 policy, neutral-head-trained fusion/lifting front end, four-channel target, split, loss and evaluation rules.

The final quantitative cohort is 58 healthy VSD knees and 13 Ruikar fractured knees. VSD z057 Left and Right are excluded for metal artefacts. VSD 010 remains included after merge recertification. Ruikar Cases 4, 8 and 10 remain excluded.

## Data contract

- Canonical world and stored-image convention: LPS.
- Intermediate CT resampling: 0.5 mm isotropic.
- Fixed physical field of view: 200 mm cubed.
- Stored target grid: 256 cubed at 0.78125 mm isotropic.
- Target order: femur, tibia, patella, fibula.
- Primary target: four independent binary occupancy channels.
- Legacy whole-bone binary and TSDF targets are forbidden in final configurations.
- Raw Regen data is not changed. Generated outputs use study IDs.

## Evaluation isolation

Exactly five subject-grouped folds are generated with seed 42. Bilateral knees remain together. For each test fold, the next fold is validation and all others are training. FCMAE, cross-view and neutral-front-end gradients may use only training subjects. Test subjects are never accessed during checkpoint selection.

## HPC execution contract

Heavy preprocessing and modelling may run on Open OnDemand. A submitted job is never considered successful without a returned result bundle and independent review. The user returns the executed notebook or logs, configuration, summary data, QA figures, resource use, output paths and hashes. The responsible agent records PASS, RETRY or BLOCKED before dependent work resumes.

## Foundation gates

- **Stage 0:** reusable agent registry, consistent configs, repository inventory and legacy quarantine.
- **Stage 1:** 71 version-matched samples, LPS geometry, approved laterality, manual fracture-ROI status and zero leakage.
- **Stage 2:** fold-isolated FCMAE/front-end checkpoints, byte-identical shared features, fair decoder definitions, metric tests, one/two-case Dice at least 0.90 per bone and successful HPC save/resume preflight.

Full five-fold decoder training is outside this specification and starts only after independent Stage 2 approval.