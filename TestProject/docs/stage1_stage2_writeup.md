# Stage 1 & Stage 2 Write-Up — 3D U-Net vs V-Net Knee Reconstruction

**Project:** Fracture-Preserving 3D Knee Reconstruction from Bi-planar X-rays: A Controlled Comparison of U-Net-style and V-Net-style Decoder Blocks
**Document purpose:** A faithful, detailed record of what was built, decided, and verified across Stage 1 (data foundation) and Stage 2 (modelling foundation), written to scaffold the capstone report. Prepared 2026-07-16.

> **Status headline (read first).** Stage 1 and Stage 2 are **built and internally validated but not yet certified or executed on real data.** The canonical manifest currently reports `certification_approved = false`, `ready_rows = 0`, `pending_recertification_rows = 71`, and every Stage 2 modelling notebook ships with `RUN_REAL_DATA = False`. There are therefore **no empirical U-Net-vs-V-Net accuracy numbers yet.** The "results" reported here are _validation, QA, and readiness_ results — geometry audits, laterality audits, cohort assembly, architecture-fairness contracts, and metric correctness tests. This distinction should be preserved in the capstone: Stage 1/2 establish a defensible, leakage-free experimental apparatus; the comparative results themselves belong to Stage 3.

---

## 1. Scientific framing and the foundation contract

The study is a **controlled decoder comparison**. Both models share an identical front end — a ConvNeXtV2 encoder, self-supervised pre-training, bi-planar cross-view fusion, and a geometry-locked 2D→3D lift — and differ in exactly one factor: the decoder block style.

- **Arm A — "plain U-Net style"**: double 3×3×3 conv → GroupNorm → ReLU blocks.
- **Arm B — "residual V-Net style"**: residual blocks with PReLU activations and a 1×1×1 projection shortcut.

Everything else (features, targets, loss, optimiser, schedule, seed, output head, wiring) is held byte-for-byte identical so that any measured difference is attributable to the decoder alone.

**Authoritative contract** (`configs/baseline_protocol_v1.json`, `docs/superpowers/specs/2026-07-12-foundation-implementation-v1.md`):

| Item                         | Value                                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------- |
| Final quantitative cohort    | **71 knees** = 58 healthy (VSD) + 13 fractured (Ruikar)                                                  |
| Reconstruction target        | **4 independent binary occupancy channels**, order `[femur, tibia, patella, fibula]`                     |
| World / stored convention    | **LPS** end-to-end                                                                                       |
| Intermediate CT resampling   | 0.5 mm isotropic                                                                                         |
| Fixed physical field of view | 200 mm cube                                                                                              |
| Stored target grid           | 256³ at 0.78125 mm isotropic                                                                             |
| Cross-validation             | 5 subject-grouped folds, seed 42, `StratifiedGroupKFold` on `subject_id`                                 |
| Validation rule              | validation fold = `(test_fold + 1) mod 5`                                                                |
| Loss                         | 0.5·BCE + 0.5·soft-Dice                                                                                  |
| Normalisation                | GroupNorm (8 groups) — BatchNorm forbidden                                                               |
| Forbidden                    | whole-bone binary union targets, TSDF targets, fine-tuned front end as the primary comparison, fold id 5 |

Two forbidden legacy target formats (binary union, TSDF) and a fine-tuned-front-end regime are explicitly barred so the comparison cannot silently drift. Regen (clinical X-rays) is **qualitative only** — no paired CT, no quantitative target — and its raw PII is never modified; all generated artefacts use study IDs.

---

## 2. Data exploration

Three datasets, three roles. Exploration notebooks live in `notebooks/data_exploration/`.

### 2.1 VSD (healthy) — quantitative

- **Two source populations**: 11 "regular" knee-region subjects (8 single-folder DICOM + 3 multi-folder that require merging) and ~20 "z-prefix" full-body CT subjects (z001–z066).
- Scans are **bilateral** → left/right legs separated by connected-component analysis; knee crop uses a ±100 mm margin around the knee centre.
- Multi-folder subjects (010, 015, 017) are merged (`vsd_multi_merge.ipynb`) before cropping; VSD010's merge geometry was independently re-certified (slice ordering, overlap/gap, affine continuity).
- After exclusions the healthy contribution is **58 knees** (bilateral, so most subjects contribute two).

### 2.2 Ruikar (fractured) — quantitative

- 16 source cases; **already knee-region** scans, so no bilateral separation is needed.
- **Bimodal Z-spacing**: 0.7 mm (8 cases) and 3.0 mm (6 cases) — noted so downstream resampling is spacing-aware.
- After exclusions: **13 usable fractured knees.**

### 2.3 Regen (clinical) — qualitative

- 30 patient folders with real bi-planar X-rays (`.dcm`) and annotations (`.ann`), ~2 views each (AP + lateral).
- No CT ground truth → **visual assessment by clinicians only**; does not enter the quantitative pipeline.
- Contains patient names → handled as PII; all outputs de-identified to study IDs.

### 2.4 Exclusions and their justifications (the decisions that shaped the cohort)

| Case(s)                         | Decision                           | Reason                                                                                                                                                                             |
| ------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VSD z057 Left & Right           | Excluded                           | Metal artefacts saturate HU                                                                                                                                                        |
| VSD z050 / z063 — one knee each | Excluded (the anatomical **Left**) | Total knee replacement; metal masquerades as bone and occludes the joint (unrecoverable). Surviving knee is anatomical **Right**, confirmed by intrinsic LPS fibula/tibia evidence |
| Ruikar Case4, Case10            | Excluded                           | Single-slice scout series (below `MIN_Z_SLICES`)                                                                                                                                   |
| Ruikar Case8                    | Excluded                           | Lodged fracture-fixation metal hardware hides bone shape                                                                                                                           |
| VSD010                          | **Retained**                       | Merge geometry re-certified; four-bone audit passes both knees                                                                                                                     |

These are the numbers behind "58 + 13 = 71": healthy 60-ish candidate knees minus TKR/metal exclusions → 58; fractured 16 minus 2 scouts and 1 metal → 13.

---

## 3. Data pre-processing

Two deterministic pipelines feed the model, plus a per-bone ground-truth build.

### 3.1 Pre-DRR CT standardisation (`predrr_preprocessing.ipynb`)

Uniform pipeline applied to all 71 volumes:

```
Load → Resample (0.5 mm iso) → Orient (LPS) → Bone window [-450, 1050] HU
     → Body-envelope mask (drop CT table + stray body parts)
     → ROI bone crop → Center into fixed 200 mm cube FOV
     → Resize to 256³ → save NIfTI (float [0,1])
```

Key cleaning steps, each a deliberate decision:

- **Body-envelope mask (all cases):** keep the largest soft-tissue connected component and fill holes, zeroing the CT table/support slab and disconnected extra body parts. This automatically cleans air-gap-separated table cases (e.g. Case5, Case11, Case12) with no manual work.
- **Fused cast (Case3, Case16):** a dense cast is topologically fused to the limb — no threshold or connected-component can separate it — so it was removed manually in 3D Slicer, de-specked (`fractured_cast_cleanup.ipynb`), and those cases re-pointed at the cleaned NIfTI.
- **Contralateral-leg contamination (VSD z036):** bilateral FOV captured opposite-leg bone that survives thresholding; soft-tissue masking cannot split one soft envelope, so erosion-based leg separation was applied (`z036_contralateral_cleanup.ipynb`).
- **Non-destructive laterality:** canonical output IDs come from an override config; **raw files are never renamed.** The active override map is empty because the focused four-bone audit found no override was required.

### 3.2 DRR generation (`drr_generation.ipynb`, DiffDRR)

- Input: the 256³ / 0.78125 mm / LPS / float[0,1] predrr volumes.
- Renders **one AP and one LAT** view per knee → 71 knees → **142 DRRs**, saved as lossless `.npy` (model input) plus 16-bit PNG (QA).
- Geometry: **SDD = 1000 mm, SOD = 850 mm** (clinical cone-beam, ~1.18× magnification), detector spacing `delx = 1.4 mm`, output 256×256 (1:1 with the volume).
- Density subtlety: DiffDRR classifies air/soft/bone by **HU** thresholds, so the normalised [0,1] intensities are first reversed back to pseudo-HU; otherwise every voxel reads as soft tissue and the bone-attenuation multiplier does nothing. This was validated on a CPU smoke test (regular VSD, z-prefix VSD, and two fractured cases) before the GPU batch.

### 3.3 Per-bone ground truth (`modeling/00_gt_per_bone.ipynb`)

Manual Slicer STLs (femur/tibia/patella/fibula) are voxelised onto each case's predrr grid and written as per-bone `.nii.gz`, then alignment-checked against the CT before modelling. This produces the **4-channel binary occupancy** target the whole study reconstructs.

---

## 4. Cohort assembly and evaluation isolation (Stage 1 core)

The canonical manifest (`00_manifest_and_reconciliation.ipynb` → `quantitative_manifest_v1.csv`) builds the 71-knee planning cohort with explicit excluded rows, five subject-grouped folds, and leakage checks.

**Leakage controls that passed (all zero):**

- no subject appears in more than one test fold,
- no rows land in the forbidden fold 5,
- every augmentation child inherits its parent's subject and fold (no augmentation leakage).

**Bilateral integrity:** left+right knees of the same subject are grouped so they never split across train/val/test. FCMAE, cross-view, and front-end gradients may touch **training subjects only**; test subjects are never opened during checkpoint selection.

**Why certification is still BLOCKED (honest status):** rows remain `pending_recertification` until (a) versioned predrr, four-bone targets, and AP/LAT DRRs come back with returned **HPC evidence**, (b) the two TKR survivor sides are user-approved, and (c) every manual fracture-ROI overlay is visually approved. File existence is explicitly _not_ accepted as evidence of success.

---

## 5. Stage 1 verification results (what actually passed)

These are the concrete Stage 1 outcomes — audits and gates, not model accuracy.

| Gate / audit                          | Notebook | Result                                                                                                                                                       |
| ------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Canonical manifest + folds + leakage  | 00       | 71 rows, 7 excluded, **zero leakage**; certification still pending                                                                                           |
| Manual fracture-ROI replay            | 01       | 13/13 cases pass geometry + retention; 5/5 synthetic replay checks pass; **visual approval still pending** (12 `verified_fracture`, 1 `no_visible_fracture`) |
| LPS geometry validation               | 02       | Contract validator in place; awaits versioned HPC batch                                                                                                      |
| Local/HPC notebook parity             | 03       | **PASS** — registered preprocessing pairs are byte-identical outside tagged config cells                                                                     |
| Stage 0 contract / legacy quarantine  | 04       | **PASS** (static config gate)                                                                                                                                |
| VSD010 merge + z050/z063 laterality   | 05       | **PASS** — merge geometry and bilateral coverage good; survivors are anatomical Right                                                                        |
| Cohort laterality (superseded X-only) | 07       | Flagged z023 Left / z036 Right — later shown to be false positives                                                                                           |
| Focused four-bone chirality audit     | 08       | **PASS — 58/58** knees; z023/z036 flags were false positives; no mapping override justified                                                                  |
| Artifact inventory / quarantine       | 06       | 405 artefacts classified, 34 legacy items quarantined (non-selectable)                                                                                       |

**Methodological highlight worth writing up:** the laterality verdict evolved from a naïve _absolute-LPS-X_ rule (not rotation-stable) to a _four-bone local chirality_ test using femur/tibia/patella/fibula centroids, which is stable under transverse-plane rotation and is cross-checked independently in both the source STL and the voxel target. This is a good example of a diagnostic being replaced when its assumption was found unsafe.

---

## 6. Encoder design (Stage 2.1 — `01_encoder_pipeline.ipynb`)

A **fold-isolated, two-phase self-supervised encoder** built on ConvNeXtV2, initialised from the pinned `convnextv2_tiny.fcmae` self-supervised weights.

- **P1 — FCMAE pre-training:** canonical uniform **60% masking**, fully self-supervised. No target, no fracture label, no cohort sampler, and no structure-derived mask ever enters P1 — this keeps pre-training honest and independent of the downstream task.
- **P2 — bidirectional cross-view completion:** adds AP↔LAT completion on top of the retained FCMAE loss, teaching the encoder to reason across the two orthogonal views.
- **Fold isolation:** five encoders, one per fold, each trained on that fold's **training subjects only**; test-fold files are never opened. Training augmentation is deterministic and **photometric only** (gamma, brightness, mild Gaussian noise), with geometric augmentations — crop, rotation, flip, elastic, cutout — explicitly forbidden so bi-planar geometry is preserved. Validation always uses clean DRRs.

The encoder is the shared, frozen "trunk" that both decoders inherit; nothing here differs between the U-Net and V-Net arms.

---

## 7. Shared front end (Stage 2.2 — `02_frontend_pretrain.ipynb`)

Between encoder and decoder sits a **neutral shared front end**: hybrid bi-planar fusion → orthogonal 3D lift, trained through a _disposable_ head that is then discarded and frozen.

**Architecture (from the code):**

- **Fusion per feature level** (`FUSION_TYPES = [local, local, attention, attention]`): shallow levels use a `LocalFusion` (concat + 3×3 conv + residual); deep levels use `CrossAttention` (learned Q/K/V across AP and LAT). Fusion is **bidirectional** — AP refined by LAT _and_ LAT refined by AP.
- **Geometry-locked orthogonal lift** (`_orthogonal_lift`): each fused 2D feature is projected to `[64,128,256,512]` channels, then back-projected into a cube — AP broadcast along one world axis, LAT (flipped) along the orthogonal axis — and fused with a 3×3×3 conv. This produces a 4-level 3D feature pyramid at `64³, 32³, 16³, 8³`. This is the LPS-consistent lift that replaced an earlier lower-quality extrusion.
- **Neutral head:** a symmetric decoder-shaped head with `single conv → GroupNorm(8) → ReLU` blocks and trilinear up-steps to 256³. It exists only to give the front end a training signal; its parameters are **discarded** so neither decoder arm inherits head-specific tuning.
- **Fair, reproducible export:** the frozen front end is verified to produce **deterministic, byte-identical features** (double forward pass, SHA-256 compared) at fixed shapes; features + targets are cached (fold 0) for the decoder gate. Optimiser Adam, lr 1e-4, 0.5·BCE + 0.5·soft-Dice loss, AMP + activation checkpointing for memory.

The result is a single frozen feature source both decoders consume identically — the linchpin of a fair comparison.

---

## 8. Decoder design and fairness (Stage 2.3 — `03_decoder_pipeline.ipynb`)

Both decoders share the same `Decoder3D` skeleton — three `ConvTranspose3d` up-steps with skip concatenation, then trilinear refinement to 128³→192³→256³, then a 1×1×1 conv to 4 output channels. **Only the block type differs:**

- **`PlainDoubleConv` (U-Net):** `Conv3d → GroupNorm(8) → ReLU → Conv3d → GroupNorm(8) → ReLU`.
- **`ResidualVNetBlock` (V-Net):** `Conv → GroupNorm → PReLU → Conv → GroupNorm`, added to a 1×1×1-projected shortcut, then PReLU.

**Enforced fairness contract (`architecture_contract`)** — the notebook refuses to run unless:

- no BatchNorm exists in either arm (GroupNorm only),
- each arm has exactly 12 GroupNorm layers,
- both arms expose the identical wiring points (`up3/up2/up1/refine128/192/256/output`).

**Capacity ("overfit") gate:** on 1-case and 2-case subsets, each arm must reach **≥ 0.90 hard Dice on every bone of every case**, for a required number of consecutive evaluations. This proves each decoder has enough capacity to fit the 256³ target before any expensive cross-validation is attempted. The gate also runs a **save/resume preflight**: at update 5 it checkpoints, reloads, and asserts parameter-hash and pre-update-logit-hash equality — guaranteeing HPC jobs can resume bit-exactly after interruption. A ≥10% GPU-headroom check guards against OOM at scale.

Because the gate depends on the fold-0 cached frozen features and requires a CUDA device, it is intended to run on the HPC GPU and is currently gated behind `RUN_REAL_DATA = False`.

---

## 9. Metrics and testing methodology (Stage 2 — `06_metric_foundation.ipynb` + `tests/`)

A single metric core is shared by the decoder, comparison, and test notebooks, with synthetic analytically-checkable tests.

**Per-bone metrics:**

- **Overlap:** Dice and IoU, with explicit handling of empty predictions (Dice 0, flagged) and invalid/empty targets (NaN, flagged) rather than silent division errors.
- **Physical surface distances:** **HD95** and **ASSD** in millimetres, computed from eroded surface masks with **spacing-aware** Euclidean distance transforms (anisotropic sampling respected). Verified against closed-form answers: perfect overlap → 0, empty → NaN, a known voxel offset → exact mm, anisotropic diagonal → exact mm.

**Fracture-aware diagnostics** (directly relevant to the conditional stretch goal):

- **Connected-component agreement** between prediction and target,
- **`false_bridge`** flag when a fractured target has ≥2 components but the prediction fuses them into fewer — i.e. the model "heals" a fracture it should have preserved,
- **minimum component-gap (mm)** between the two largest fragments.
  Synthetic tests confirm a genuine two-fragment fracture is detected, an artificial bridge collapses the component count and raises `false_bridge`, and the known 5 mm gap is measured exactly.

**Aggregation:** metrics aggregate **subject-first** (bilateral knees of one subject averaged together) before the cohort mean, so subjects with two knees are not double-counted.

**Planned statistics (Stage 3, not yet run):** paired **Wilcoxon signed-rank** tests across the five folds to compare U-Net vs V-Net, per the fracture-comparison methodology (cross-validation + surface metrics + non-parametric paired test). `04_decoder_comparison.ipynb` currently validates only the Stage 2 schema; full five-fold aggregation and thesis comparisons are explicitly deferred to Stage 3.

---

## 10. HPC execution and evidence discipline

Heavy preprocessing and modelling run on the Sunway HPC (AWS Linux via Open OnDemand, CUDA, TotalSegmentator full model). A **local digital twin** mirrors the HPC layout, and a parity gate enforces byte-identical algorithm cells between local and HPC notebooks (only `environment-configuration`-tagged cells may differ).

The handoff rule is strict and worth stating in the report as a methodological strength: **a submitted job is never treated as successful without a returned result bundle** (executed notebook/logs, config, summary CSV/JSON, QA figures, resource usage, output paths, and hashes of large outputs) **plus an independent PASS review.** Dependent work resumes only after a recorded PASS.

---

## 11. Current status summary

| Layer                                        | Built         | Internally validated                       | Executed on real data                      | Certified                                    |
| -------------------------------------------- | ------------- | ------------------------------------------ | ------------------------------------------ | -------------------------------------------- |
| Stage 1 data foundation                      | ✅            | ✅ (geometry, laterality 58/58, leakage 0) | Local preprocessing yes; HPC batch pending | ❌ (pending HPC evidence + visual approvals) |
| Stage 2.1 encoder (FCMAE P1 / cross-view P2) | ✅            | ✅ (contract + isolation logic)            | ❌ (`RUN_REAL_DATA=False`)                 | ❌                                           |
| Stage 2.2 shared front end                   | ✅            | ✅ (deterministic feature hashing)         | ❌                                         | ❌                                           |
| Stage 2.3 decoders + overfit gate            | ✅            | ✅ (fairness contract, save/resume logic)  | ❌ (needs HPC GPU)                         | ❌                                           |
| Stage 2 metric core + tests                  | ✅            | ✅ (synthetic analytic tests)              | n/a                                        | ❌ (Agent-N review pending)                  |
| Stage 3 five-fold comparison + Wilcoxon      | Designed only | —                                          | ❌                                         | Blocked until Stage 2 PASS                   |

**In one sentence:** the experimental apparatus for a rigorous, leakage-free U-Net-vs-V-Net comparison is complete and self-consistent; the empirical comparison has not yet been run.

---

## 12. Suggested mapping to capstone report chapters

- **Introduction / Problem** → §1 (framing, why a _controlled_ decoder comparison).
- **Literature / Methodology background** → §6–§9 (FCMAE, cross-view fusion, U-Net vs V-Net, surface metrics + Wilcoxon).
- **Datasets** → §2 (three datasets, roles, exclusions with justification).
- **Data preparation** → §3–§4 (preprocessing, per-bone GT, cohort + fold design).
- **Experimental design / rigour** → §1 fairness contract, §4 leakage control, §8 architecture contract, §10 evidence discipline. _This is the chapter where the project is strongest and most defensible — lead with it._
- **Results** → §5 (Stage 1 validation results). Be explicit that model-accuracy results are Stage 3 and label them as future work; present the four-bone chirality correction (§5) as a genuine finding.
- **Evaluation plan / Future work** → §9 metrics, §11 status, the conditional fracture-aware stretch goal, and the pending HPC run.

### Honesty note for the report

Do not present placeholder or expected Dice/HD95 numbers as findings. The credible Stage 1/2 contribution is the _methodology and validated foundation_ (balanced healthy/fractured cohort, LPS geometry, four-bone laterality verification, fold isolation, a provably fair shared front end, and fracture-sensitive metrics). Frame the U-Net-vs-V-Net accuracy comparison as the immediate next step, contingent on the recorded Stage 1 PASS and the HPC GPU run.
