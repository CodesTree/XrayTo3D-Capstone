# Stage 1–4 Methodology, Justification and Results

**Project:** Comparative 3D reconstruction of knee anatomy from bi-planar radiographs using U-Net-style and V-Net-style decoders  
**Researcher:** Chan Zheng Shao, Sunway University  
**Supervisor:** Assoc. Prof. Ts. Dr Lee Yun Li  
**Evidence cut-off:** 24 July 2026  
**Purpose:** Report-ready account of the complete project workflow, from data exploration to modelling and evaluation, based on the notebooks, manifests, model provenance and QA records present in this repository.

## Executive summary and claims boundary

This study investigates whether decoder design affects the reconstruction of four knee bones—femur, tibia, patella and fibula—from paired anteroposterior (AP) and lateral (LAT) radiographs. The scientific comparison is deliberately conditional: the 2D encoder, cross-view fusion and 2D-to-3D lifting pathway are held constant within each fold, while the decoder block is varied. The original U-Net-versus-V-Net comparison was expanded to a matched \(2\times2\) architecture design because the named baselines differ simultaneously in residual topology and activation function. The four arms are therefore:

| Arm | Residual block | Activation | Parameters |
|---|---:|---|---:|
| U (ReLU, plain) | No | ReLU | 8,429,956 |
| PReLU + plain | No | PReLU | 8,430,916 |
| ReLU + residual | Yes | ReLU | 8,604,516 |
| V (PReLU, residual) | Yes | PReLU | 8,605,476 |

The current evidence supports the following conclusions:

1. **Stage 1 is complete and independently certified.** The quantitative cohort contains 71 ready knees from 43 contributing subjects: 58 healthy VSD knees and 13 fractured Ruikar knees. Seven exclusions are preserved, all versioned inputs are present, all five subject-grouped splits are leakage-free, 284 sample–bone target rows passed automated retention checks, all 13 fracture-region records were reviewed, and 142 AP/LAT DRRs passed numerical and identity QA.
2. **Stage 2 is complete with explicit qualifications.** Five fold-specific ConvNeXtV2/FCMAE encoders, cross-view P2 checkpoints and frozen shared-front-end checkpoints were produced. The combined decoder-entry gate is recorded as `PASS`, but this pass uses a project-owner override. Two plain-decoder one-case capacity runs did not achieve the predeclared 0.90 per-bone Dice criterion, and feature-quality warnings—most importantly fold 4 dimensional-collapse evidence—remain part of the provenance.
3. **Stage 3 training has been executed.** All 20 revised runs (five folds × four decoder arms) have successful run summaries, hash-recorded checkpoints, unchanged fold-specific front ends and 71 reported out-of-fold predictions per arm in aggregate. Mean best validation macro Dice across folds was 0.4174 for U, 0.4143 for plain-PReLU, 0.4635 for residual-ReLU and 0.4552 for V. These are checkpoint-selection values, not held-out comparative results.
4. **Stage 4 is not yet reproducible from the local handoff.** Only one of the expected 20 `oof_metrics.csv` files is currently present locally. Therefore, a complete 71-knee paired comparison, healthy/fractured subgroup analysis, per-bone result table and final architecture decision cannot be regenerated. No winning decoder should be claimed from validation scores or from the single available OOF file.
5. **No statistical hypothesis tests are included in the revised reporting plan.** Results are to be reported descriptively using subject-level and subgroup summaries. The earlier protocol language concerning significance tests and multiplicity correction is superseded for the thesis write-up. “Inconclusive” remains a valid outcome when overlap and boundary evidence do not consistently favour the same design.

The conditional fracture-aware architecture has not been activated. It should be considered only after the complete Stage 4 evidence demonstrates inadequate fracture-case reconstruction and identifies which baseline provides the more defensible starting point.

## 1. Research design

### 1.1 Aim

The primary aim is to compare U-Net-style and V-Net-style 3D decoders for reconstructing knee anatomy from two orthogonal radiographic views. The target is not a CT-intensity volume. It is a four-channel binary occupancy volume in the fixed order:

```text
[femur, tibia, patella, fibula]
```

This formulation preserves individual bone identity, permits per-bone evaluation and avoids allowing the large femur and tibia to conceal poor reconstruction of the smaller patella and fibula.

### 1.2 Research questions

The project addresses four linked questions:

1. Can a consistent, leakage-free and geometrically valid cohort be assembled from healthy CT, fractured CT and unpaired clinical radiographs?
2. Can a fold-specific shared front end learn useful bi-planar representations without accessing test subjects?
3. Under the same frozen front end, does residual topology or activation choice produce a consistent descriptive improvement in 3D reconstruction?
4. Are any apparent gains retained for fractured knees and thin or fracture-sensitive structures, rather than being driven only by healthy anatomy?

### 1.3 Four-stage workflow used in this report

To avoid ambiguity between notebook numbers and experimental phases, “Stage 1–4” is used as follows:

| Stage | Scope | Current status |
|---|---|---|
| **Stage 1: Data foundation** | Exploration, exclusions, CT standardisation, targets, DRRs, fracture ROIs, cohort manifest and folds | Complete; independent PASS |
| **Stage 2: Modelling foundation** | Fold-specific self-supervision, cross-view learning, feature audit, shared 3D front end, decoder capacity and QA gate | Complete with retained warnings and owner override |
| **Stage 3: Controlled decoder training** | Five-fold training of the four matched decoder arms using frozen fold-specific features | 20/20 run summaries successful |
| **Stage 4: Evaluation and interpretation** | Paired OOF aggregation, per-bone and subgroup results, robustness checks and Regen qualitative review | Incomplete local evidence; no final winner |

### 1.4 Why the comparison is controlled

Changing an encoder, fusion module, target, loss or split at the same time as the decoder would make causal attribution impossible: any observed difference could originate from several factors. Within each fold, this project therefore supplies all decoder arms with the same frozen encoder/fusion/lift state and uses the same target order, optimiser, loss, seed, output resolution and data split. The front-end state is hashed before and after every decoder run; all 20 run summaries report `front_end_unchanged = true`.

The original U-Net architecture introduced a contracting path, expanding path and skip connections to combine semantic context with localisation detail ([Ronneberger et al., 2015](https://lmbweb.informatik.uni-freiburg.de/Publications/2015/RFB15a/)). V-Net extended volumetric segmentation with residual learning and a Dice-based objective designed for foreground imbalance ([Milletari et al., 2016](https://arxiv.org/abs/1606.04797)). A direct comparison of “plain/ReLU U-Net” with “residual/PReLU V-Net” changes two factors at once. Adding the plain/PReLU and residual/ReLU arms makes the design diagnostically clearer, even though the modest sample size limits the strength of any universal architecture claim.

## 2. Data exploration

### 2.1 VSD healthy CT cohort

The VSD source comprised two acquisition groups:

- 11 regular lower-limb subjects, including eight single-folder DICOM studies and three multi-folder studies (010, 015 and 017) that required physical-position-based merging;
- 20 z-prefix full-body CT subjects identified during extended exploration.

The scans were bilateral. Each usable scan was separated into left and right legs using connected-component and anatomical checks before a knee-centred crop was generated. Eleven regular subjects contributed 22 knees. The z-prefix group contributed 36 knees after excluding both knees of z057 and one knee each from z050 and z063. The final healthy cohort was therefore 58 knees.

Exploration identified several issues that materially changed the pipeline:

- filename ordering could not be trusted for multi-folder DICOM merging; physical `ImagePositionPatient` coordinates were required;
- bilateral crops could retain contralateral anatomy when both legs shared one soft-tissue envelope;
- absolute LPS-X position was not a rotation-stable laterality rule;
- a variable cube sized from each crop would create different physical scales for healthy and fractured scans;
- metal and total knee replacement artefacts could imitate high-density bone while hiding the true joint surface.

The final laterality method used the local spatial relationship among femur, tibia, patella and fibula centroids and checked the relation in both source STL and voxelised target space. This four-bone chirality audit passed all 58 retained VSD knees. The earlier z023 and z036 flags from the absolute-X heuristic were shown to be false positives. The retained z050 and z063 knees are anatomical Right; their anatomical Left TKR knees are excluded. Raw source files were not renamed.

### 2.2 Ruikar fractured CT cohort

The Ruikar collection contained 16 source cases and was already centred on the knee, so bilateral separation was unnecessary. Exploration found a bimodal through-plane spacing: eight cases at approximately 0.7 mm and six at approximately 3.0 mm among the non-scout volumes. This variation justified using physical-coordinate and spacing-aware operations throughout resampling and evaluation.

Three cases were excluded:

- Case4 and Case10 were single-slice scout/localiser images and failed the minimum-volume requirement;
- Case8 contained fracture-fixation hardware that obscured the target anatomy.

Thirteen fractured knees remained. Case3 and Case16 were retained after manual removal of dense casts followed by automated speck cleanup. The intent was to preserve salvageable fractured anatomy instead of excluding difficult but informative cases.

### 2.3 Regen clinical X-rays

The Regen collection contains 30 folders, 60 DICOM radiographs and 90 annotation files, corresponding to 26 unique patients because four patients have repeat visits. Approximately two views are available per visit. Regen has no paired CT or four-bone 3D reference and therefore cannot contribute to quantitative training, checkpoint selection or decoder ranking.

Regen is reserved for qualitative assessment of reconstruction plausibility on real clinical X-rays. Raw names and filenames are treated as personally identifiable information. Raw data are not modified, and generated manifests, figures and handoffs must use study IDs only. This separation prevents an unpaired clinical dataset from being presented as quantitative validation.

### 2.4 Cohort yield

| Dataset | Source contribution | Retained | Excluded | Role |
|---|---:|---:|---:|---|
| VSD healthy | 62 candidate knees | 58 | 4 | Quantitative CT→DRR |
| Ruikar fractured | 16 cases | 13 | 3 | Quantitative CT→DRR |
| **Total quantitative** | **78 rows** | **71** | **7** | Training and OOF evaluation |
| Regen | 30 folders / 26 patients | Not applicable | Not applicable | Qualitative only |

The certified manifest contains 43 contributing subjects: 30 healthy subjects and 13 fractured subjects. Twenty-eight healthy subjects contribute bilateral knees; two contribute one retained knee. Subject-level grouping is consequently essential so that bilateral healthy participants do not receive twice the weight of a single-knee fractured participant.

## 3. Stage 1—Data foundation

### 3.1 CT standardisation

All quantitative CT inputs were transformed using the same physical pipeline:

```text
load source volume
→ resample to 0.5 mm isotropic
→ canonicalise to LPS
→ apply bone window [-450, 1050] HU
→ remove table and disconnected body components
→ isolate knee region
→ centre within a fixed 200 mm × 200 mm × 200 mm field of view
→ resize/store on a 256³ grid
→ normalise intensity to [0, 1]
```

The 0.5 mm intermediate grid reduces aliasing during cropping and target alignment. The stored 200 mm/256 grid has 0.78125 mm isotropic spacing. A fixed physical field of view was chosen instead of padding each volume to its own longest axis. The latter would have made long fractured scans physically coarser than healthy ±100 mm knee crops, introducing a cohort-dependent scale confound before model training.

LPS was maintained end to end because orientation errors affect laterality, AP/LAT projection geometry, target overlap and distance metrics simultaneously. Geometry was therefore treated as a data contract rather than a display preference.

The body-envelope operation retained the largest filled soft-tissue component and removed the CT table and disconnected body parts. It was sufficient for air-separated table cases. It could not separate casts fused to the limb or two legs within a shared soft-tissue envelope, which justified the limited case-specific cleanup for Case3/Case16 and z036.

### 3.2 Per-bone target construction

Four manually segmented STL surfaces were selected per case and voxelised onto the exact pre-DRR grid. The output version is `pb_occ_lps_256_v1`. Each channel is binary and non-empty. A union mask and TSDF target were explicitly quarantined because they would change the scientific task and obscure bone-specific failure.

Automated retention evidence covered:

- 71 unique samples;
- four bones per sample;
- 284 unique sample–bone rows;
- zero ROI-crop, unexpected-FOV, replay, binarity, non-empty or grid-alignment failures.

Thirty-nine fractured long-bone rows touched the superior or inferior 200 mm FOV boundary. These were expected shaft truncations, not target failures. Case3 fibula and Case13 patella each lost one 0.5 mm intermediate-grid voxel, with retention above 0.99999; the independent review judged these below anatomical significance.

### 3.3 Fracture-region annotation

The 13 retained Ruikar cases were reviewed with versioned fracture-region records. Twelve were classified as `verified_fracture`; Case2 was explicitly adjudicated as `no_visible_fracture`. All 13 records received visual approval and identity alignment. Fracture ROIs were replayed through the same spatial transform as the CT and target, preventing a source-space annotation from being compared naively with a resampled target.

This ROI layer supports fracture-sensitive diagnostics and guards against a model appearing successful by reconstructing a smooth, healthy-looking bone while closing a genuine fracture gap.

### 3.4 DRR generation

One AP and one LAT DRR were rendered for every retained CT volume using DiffDRR:

- source-to-detector distance: 1000 mm;
- source-to-object distance: 850 mm;
- nominal magnification: approximately 1.18;
- detector spacing: 1.4 mm;
- image size: 256 × 256;
- output: float32 `.npy` for modelling and 16-bit PNG for QA.

DiffDRR expresses Siddon ray tracing as vectorised PyTorch operations, allowing GPU acceleration and differentiability ([Gopalakrishnan & Golland, 2022](https://arxiv.org/abs/2208.12737)). A critical implementation correction was to reverse the normalised volume to pseudo-HU before attenuation classification. Passing values in \([0,1]\) directly into HU-based material thresholds would classify nearly every voxel as soft tissue and suppress the intended bone contrast.

The returned HPC evidence contained exactly:

- 71 cases and 142 unique views;
- 116 healthy and 26 fractured views;
- 71 AP and 71 LAT images;
- 142 NPY/PNG pairs and 142 metadata rows;
- 142 numerical-QA PASS rows and zero failures;
- 147 expected hash entries.

Independent remeasurement of ten returned HPC arrays reproduced the recorded intensity statistics, AP/LAT differences and PNG conversion. This is stronger evidence than file existence alone.

### 3.5 Manifest, folds and leakage control

The canonical manifest is `quantitative_manifest_v1`, with SHA-256:

```text
bdaef4df5e93f7ff2150051931a024147d4ead8f6f4f80caf1a230a9a590868d
```

Five folds were assigned with `StratifiedGroupKFold`, seed 42 and `subject_id` as the grouping variable. For test fold \(k\), validation fold was fixed as \((k+1)\bmod5\); the remaining folds formed training data. Fold sizes were 15/14/14/14/14 knees. Fractured counts were 3/2/2/3/3 and healthy counts were 12/12/12/11/11.

The design prevents two common forms of leakage:

1. left and right knees of a participant cannot appear in different splits;
2. encoder pretraining and cross-view learning cannot use a test participant merely because those stages are self-supervised.

All recorded leakage counts were zero: subject across test folds, forbidden fold 5, augmentation-parent mismatch, and train/validation/test subject overlap in every fold.

### 3.6 Stage 1 results and interpretation

| Gate | Result |
|---|---|
| Manifest state | 78 rows: 71 ready, 0 pending, 7 excluded |
| Cohort contract | 58 healthy + 13 fractured |
| Per-bone target rows | 284/284 passed automated retention |
| VSD laterality | 58/58 passed focused four-bone audit |
| Fracture-region review | 13/13 approved; 12 verified, 1 no-visible-fracture |
| DRR batch | 142/142 numerical QA PASS |
| Fold leakage | Zero on all tested axes |
| Independent Stage 1 verdict | PASS, 16 July 2026 |

Stage 1 therefore provides a defensible quantitative cohort with consistent geometry and explicit exclusions. Its main methodological contribution is not a model accuracy result; it is the removal of preventable confounds involving laterality, physical scale, DICOM ordering, bilateral leakage, metal artefacts and fracture-region retention.

## 4. Stage 2—Modelling foundation

### 4.1 Fold-specific encoder policy

Five ConvNeXtV2-Tiny encoders were created, one for each test fold. Each was initialised from the pinned public `convnextv2_tiny.fcmae` checkpoint, whose source, licence and loaded-state hash are recorded. ConvNeXt V2 was co-designed with a fully convolutional masked-autoencoder strategy and Global Response Normalization to support masked self-supervised learning ([Woo et al., 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html)).

The canonical executed pipeline uses FCMAE P1 followed by cross-view P2. The unexecuted notebooks under `HPC/HPC_notebooks/modeling_HPC_new/` describe a separate fixed-split SimCLR/joint-training experiment with `RUN_TRAINING = False`; they are not evidence for the results reported here and should not be mixed with the canonical five-fold methodology. SimCLR is a defensible future alternative—its original study emphasises augmentation composition and a nonlinear projection head ([Chen et al., 2020](https://proceedings.mlr.press/v119/chen20j))—but it has not replaced the executed FCMAE/P2 experiment.

### 4.2 Phase P1: FCMAE

P1 reconstructed uniformly masked DRR patches and did not use bone targets, fracture labels or cohort-balanced sampling. Key parameters were:

- mask ratio 0.60;
- patch size 32;
- decoder dimension 512;
- micro-batch 16 with eight-step accumulation;
- effective batch 128;
- maximum 250 epochs;
- peak learning rate \(7.5\times10^{-5}\);
- weight decay 0.05;
- warm-up fraction 0.08.

Uniform masking maintained a genuinely self-supervised objective. A fracture-aware mask during P1 would leak downstream anatomical priorities into the generic representation and would make the baseline more difficult to interpret.

### 4.3 Phase P2: bidirectional cross-view completion

P2 retained the FCMAE objective and added AP↔LAT completion. Its purpose was to make the encoder sensitive to complementary information across orthogonal views instead of treating every DRR as an unrelated 2D image. The maximum duration was 100 epochs, with the remaining optimiser and effective-batch settings matched to P1.

Only photometric training augmentation was allowed:

- gamma 0.9–1.1;
- brightness offset −0.05 to 0.05;
- Gaussian noise \(\sigma=0\)–0.02;
- clamp to \([0,1]\);
- independent perturbations for AP and LAT.

Crop, rotation, translation, flip, elastic deformation, cutout and random erasing were prohibited because they would alter the calibrated AP/LAT relationship. Clean validation DRRs were used for checkpoint selection.

### 4.4 Encoder results

| Fold | P1 epochs | Best P1 validation loss | P2 epochs | Best P2 validation loss | P2 paired-vs-shuffled subject margin | P2 audit |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 250 | 0.109659 | 100 | 0.431922 | 0.0005702 | PASS |
| 1 | 250 | 0.098135 | 100 | 0.409764 | 0.0004692 | PASS |
| 2 | 250 | 0.103814 | 100 | 0.434871 | 0.0006448 | PASS |
| 3 | 250 | 0.106803 | 100 | 0.452421 | 0.0005663 | PASS |
| 4 | 250 | 0.097617 | 100 | 0.420043 | 0.0000789 | PASS |

P1 and P2 losses should not be compared directly because P2 contains an additional cross-view component. Feature standard deviations remained non-zero at the end of training, which argues against trivial total collapse. The paired-vs-shuffled margins were positive at subject level for all folds but very small, particularly fold 4. These margins support only a limited claim that correct AP/LAT pairing carried measurable information; they do not prove strong absolute 3D readiness.

### 4.5 Feature audit and retained warnings

The evaluation-only feature audit checked local linear-probe behaviour, AP/LAT retrieval and effective rank without updating model weights. The combined QA process retained the following qualifications:

- folds 0 and 3 retained P2-preservation warnings;
- fold 3 retained an L2 effective-rank caveat;
- fold 4 retained failed global-feature evidence, P2-preservation warning and L2/L3 dimensional-collapse evidence;
- fold 4 must remain in the primary comparison but must also be examined separately in a fold-4-excluded sensitivity description;
- the project must not claim that the front end is absolutely high quality merely because the decoder comparison is matched.

Weakness in a shared representation does not make inputs unequal among decoder arms within a fold, but it can depress absolute reconstruction quality and can interact with decoder design. The correct inferential boundary is therefore “decoder differences conditional on these fold-specific frozen representations.”

### 4.6 Shared bi-planar front end

For each fold, the P2 encoder was combined with:

1. local bidirectional fusion at the 64×64 and 32×32 feature levels;
2. learned cross-attention at 16×16 and 8×8;
3. a geometry-locked orthogonal lift into LPS-aligned 3D feature volumes.

Local fusion preserves fine spatial detail where full attention is expensive. Attention is used at deeper, smaller feature maps where global AP/LAT context is computationally feasible. The orthogonal lift broadcasts the AP and flipped LAT maps along their corresponding world axes and fuses them with 3D convolution. It supplies a fixed geometric inductive bias rather than asking a small dataset to learn view orientation from scratch.

The exported feature pyramid has:

```text
64 channels at 64³
128 channels at 32³
256 channels at 16³
512 channels at 8³
```

The front end was trained through four disposable multiscale 1×1×1 heads, using the four-channel occupancy target, and the heads were discarded. This prevents either final decoder from inheriting a head already specialised toward its own block style. Fixed-input repeat passes and hashes were used to verify deterministic features.

For the returned fold-0 provenance, the shared front end trained for up to 40 epochs with Adam, learning rate \(10^{-4}\), batch size 1, AMP and equal BCE/soft-Dice weighting. It achieved a best validation macro Dice of 0.5044 through its disposable supervision heads. This is a training-stage diagnostic, not an OOF reconstruction result.

### 4.7 Decoder architecture and capacity gate

Each decoder used the same three-level transposed-convolution and skip-concatenation skeleton. Four matched block definitions crossed:

- plain versus projected residual topology;
- ReLU versus PReLU activation.

Each arm contains eight GroupNorm layers with eight groups and no BatchNorm. GroupNorm is independent of batch-level statistics and is consequently appropriate for memory-limited batch-size-one 3D learning; its stability at small batches is the central motivation described by [Wu and He (2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html).

The decoder emits four logits at 128³ and then applies the same trilinear upsample to 256³ with `align_corners=False`. This reduces memory use but imposes an effective native spacing of 1.5625 mm, despite evaluation on a 0.78125 mm grid. Boundary distances below the native logit spacing must therefore be interpreted cautiously.

The capacity gate attempted to overfit:

- one healthy knee;
- a two-knee mixed subset containing one healthy and one fractured knee.

All four arms passed the two-case mixed gate and the save/resume hash check. The residual-ReLU and residual-PReLU arms passed the one-case gate. The U and plain-PReLU arms failed because patella/fibula Dice remained below 0.90 despite 350 epochs:

| One-case arm | Femur | Tibia | Patella | Fibula | Gate |
|---|---:|---:|---:|---:|---|
| U (ReLU, plain) | 0.934 | 0.955 | 0.764 | 0.781 | FAIL |
| PReLU + plain | 0.954 | 0.953 | 0.804 | 0.844 | FAIL |
| ReLU + residual | 0.945 | 0.917 | 0.968 | 0.944 | PASS |
| V (PReLU, residual) | 0.944 | 0.907 | 0.914 | 0.914 | PASS |

The project owner authorised Stage 3 for comparative evaluation while preserving these failures. The combined verdict is a procedural `PASS` with `ACCEPT_WITH_PROJECT_OWNER_OVERRIDE`; it must not be rewritten as eight successful capacity runs. This finding already suggests that residual topology may make optimisation of small structures easier, but it is not a held-out performance result.

### 4.8 Metric foundation

The metric notebook defines:

- Dice and IoU for overlap;
- ASSD and HD95 in millimetres for surface error;
- prediction and target connected-component counts;
- empty-prediction flags;
- a false-bridge indicator for a predicted component that closes a target fracture gap.

Spacing-aware Euclidean distance transforms are used for surface distances. Empty valid targets are treated as invalid and flagged; a missing prediction against a valid target receives Dice 0 rather than disappearing from a macro average. Boundary-censored handling is required where long bones meet the fixed FOV cut plane.

Dice alone is insufficient because a volumetric overlap can remain acceptable while a joint surface, thin fibula or fracture boundary is misplaced. Problem-aware metric selection and explicit handling of metric pitfalls are consistent with the recommendations of [Metrics Reloaded](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182665/). Disaggregated healthy/fractured and per-bone reporting is also supported by the closest broad benchmark of bi-planar X-ray-to-3D bone reconstruction, which warns that pooled results may overestimate performance in clinically relevant subgroups ([Shakya and Khanal, 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html)).

All 25 synthetic analytic metric checks passed.

## 5. Stage 3—Controlled decoder training

### 5.1 Training protocol

The revised authoritative training notebook is `03b_decoder_cross_validation.ipynb`. For every fold and arm:

- the fold-specific front end was loaded and frozen;
- decoder weights began from the same hash-recorded shared initialisation;
- batch size was 1;
- optimiser was Adam with learning rate 0.001;
- scheduler was cosine annealing;
- maximum duration was 20 epochs;
- early-stopping patience was 15;
- AMP and activation checkpointing were enabled;
- seed was 42;
- threshold for hard occupancy was 0.5;
- loss was \(0.5\times\) training-split-balanced BCE \(+0.5\times\) soft Dice.

Per-bone positive weights were computed from the current fold’s training targets only. This addresses severe voxel imbalance, especially for patella and fibula, without using validation or test-target prevalence. The weights varied slightly by fold but were approximately 35–37 for femur, 51–52 for tibia, 440–443 for patella and 570–573 for fibula.

Earlier unweighted pilot decoder runs were explicitly excluded after the protocol amendment. They must not be combined with the revised weighted runs.

### 5.2 Execution results

All 20 run summaries report:

- `success = true`;
- 20 completed epochs;
- correct OOF row count for the fold;
- a checkpoint and configuration SHA-256;
- unchanged front-end state;
- seed 42 and revised weighted loss;
- approximately 64% GPU headroom on a 23.67 GB device.

Best validation macro Dice was:

| Fold | U (plain/ReLU) | Plain/PReLU | Residual/ReLU | V (residual/PReLU) |
|---:|---:|---:|---:|---:|
| 0 | 0.4465 | 0.4360 | **0.4711** | 0.4621 |
| 1 | 0.4413 | 0.4375 | **0.4790** | 0.4777 |
| 2 | 0.3816 | 0.3762 | **0.4313** | 0.4226 |
| 3 | 0.4275 | 0.4281 | **0.4948** | 0.4781 |
| 4 | 0.3901 | 0.3939 | **0.4412** | 0.4355 |
| **Mean** | **0.4174** | **0.4143** | **0.4635** | **0.4552** |

The residual-ReLU arm had the highest best-validation Dice in every fold, and both residual arms exceeded the plain arms in the fold means. This is a consistent training-stage pattern and agrees with the capacity-gate behaviour. It is nevertheless insufficient to declare a winner because:

- validation data selected the checkpoints;
- the values are not paired held-out OOF outcomes;
- validation fold composition changes across the five training configurations;
- Dice alone does not test boundary accuracy or fracture preservation;
- fold 4 has known front-end quality concerns;
- only one training seed was used.

Mean wall time per arm/fold was approximately 37.5–38.3 minutes. Peak GPU use was about 8.4 GB, leaving substantial recorded headroom. Residual arms contain only about 175,000 more parameters than plain arms, so the resource difference was small relative to the shared 3D workload.

### 5.3 Stage 3 interpretation

Stage 3 demonstrates that the revised protocol can train all matched models without changing their shared front ends. It also produces a preliminary optimisation finding: residual topology was associated with stronger one-case memorisation and higher checkpoint-selection Dice, while changing ReLU to PReLU alone did not consistently improve the plain decoder. These observations motivate, but do not replace, the held-out paired evaluation.

## 6. Stage 4—Evaluation and results interpretation

### 6.1 Intended quantitative evaluation

The expected OOF evidence is \(71\) knees × \(4\) arms = \(284\) paired prediction rows, with every knee evaluated only by the model for which its fold was held out. Evaluation should proceed in this order:

1. validate run hashes, manifest hash, checkpoint identity, fold identity and frozen-front-end identity;
2. verify exactly 71 unique knees for every arm;
3. compute per-bone Dice, IoU, ASSD, HD95, empty predictions, component counts and false bridges;
4. average bilateral knees within each subject first;
5. report subject-level arm summaries for all 43 subjects;
6. report healthy and fractured subgroups separately;
7. report each bone separately, with particular attention to patella and fibula;
8. describe fold-4-excluded results as a sensitivity check because fold 4 has retained front-end collapse warnings;
9. review de-identified Regen reconstructions qualitatively, without using them to select an arm.

Subject-first aggregation is essential. A knee-first average would allow a bilateral healthy participant to contribute twice the weight of a fractured participant, contrary to the study’s fracture-sensitive aim.

### 6.2 Revised non-inferential reporting

Statistical hypothesis tests are not part of the final methodology. In particular, the thesis should not report p-values, Wilcoxon tests, permutation significance, ANOVA-style tests or multiplicity-adjusted significance decisions for the decoder comparison.

The final tables should be descriptive:

- number of subjects and knees;
- mean and median subject-level Dice and ASSD by arm;
- per-bone mean and median metrics;
- healthy and fractured subgroup summaries;
- empty-prediction and false-bridge counts;
- fold-level values to show consistency;
- resource and parameter comparisons.

The architecture interpretation should require agreement in practical direction across overlap and surface quality. If one arm improves Dice but worsens ASSD, or if an apparent advantage disappears for fractured cases, the result should be described as mixed or inconclusive. This is a scientific conclusion, not a failure to obtain significance.

The older protocol-amendment text contains significance-testing and Holm-correction language. That analysis rule is superseded by the present no-test reporting decision. The data isolation, hash acceptance, fold-4 sensitivity and claims-boundary portions of the amendment remain relevant.

### 6.3 Current Stage 4 evidence

The local repository contains run summaries for all 20 Stage 3 runs but only:

```text
models/decoders/foundation_stage2_v1/
  fold_0/plain_prelu_style/oof_metrics.csv
```

That file contains 15 fold-0 knees for one arm. It is useful for checking the metric schema, but it cannot support an architecture comparison. Nineteen expected OOF metric files and their complete prediction-mask handoff are absent from the local evidence tree. The `reports/` tree also does not contain a completed `comparison_summary.json` or all-arm subject-level comparison tables.

Consequently, the following results are **not yet available**:

- final OOF Dice/ASSD/HD95 for any complete arm;
- a paired U-versus-V result across 43 subjects;
- residual and activation descriptive effects;
- per-bone all-arm rankings;
- complete healthy-versus-fractured comparison;
- fold-4-excluded sensitivity result;
- false-bridge comparison;
- qualitative clinician verdicts for Regen;
- evidence-based decision on the fracture-aware stretch goal.

The correct Stage 4 result at this evidence cut-off is therefore:

> The comparative evaluation remains pending local return and hash verification of all fold/arm OOF metrics and masks. No decoder is declared superior.

### 6.4 Required HPC result handoff

Before the final results chapter can be completed, the HPC return should include:

- executed `03b_decoder_cross_validation.ipynb` or complete run logs;
- executed `04_decoder_comparison.ipynb`;
- the exact configuration and manifest hash;
- all 20 `oof_metrics.csv` files;
- all required OOF prediction masks or a documented, verified storage location;
- all 20 `run_summary.json` and checkpoint hashes;
- comparison summary CSV/JSON files and QA figures;
- resource usage and job/output paths;
- hash manifest for large files;
- a recorded PASS/RETRY verdict after local reconciliation.

Submission or the existence of checkpoint files alone is not evidence that the comparative result is complete.

## 7. Integrated discussion

### 7.1 Methodological strengths

The strongest aspect of the project is its control of non-architectural confounds. The same four-channel target, physical FOV, LPS convention, view geometry, subject-grouped folds, loss, seed and fold-specific front end are shared across arms. The project also preserves negative findings: false laterality flags were corrected rather than hidden, capacity failures were not relabelled as passes, and fold 4 warnings remain visible after owner-authorised continuation.

The cohort includes both healthy and fractured anatomy, and fractured cases are not treated as a late external test only. This matters because a model can produce plausible healthy shapes while smoothing fracture discontinuities. Per-bone targets, surface distances, component diagnostics and false-bridge checks make that failure observable.

### 7.2 Limitations

The main limitations are:

- only 43 contributing subjects and 13 fractured subjects;
- a single training seed for the primary runs;
- synthetic DRRs for quantitative testing, which do not reproduce every characteristic of real radiographs;
- no 3D ground truth for Regen;
- conditional inference on front ends with retained P2 warnings;
- fold 4 global-feature and dimensional-collapse concerns;
- project-owner override of two failed one-case capacity gates;
- 128³ native logits upsampled to 256³, limiting genuine boundary resolution;
- substantial class imbalance despite training-split positive weighting;
- incomplete local Stage 4 OOF evidence at the current cut-off;
- no clinical-utility or deployment claim.

The five-fold design makes efficient use of a small cohort and ensures every knee can receive a held-out prediction, but it does not create five independent datasets. Fold results should therefore be presented as consistency information rather than as five separate replications.

### 7.3 Conditional fracture-aware extension

A fracture-aware model is justified only if the complete Stage 4 subgroup results show clinically meaningful weakness in fractured cases, such as:

- materially worse fractured-case Dice and ASSD than healthy-case values;
- frequent false bridges or component loss in verified fracture ROIs;
- systematic smoothing of fracture gaps in qualitative overlays;
- disproportionate failures for the fibula, patella or fracture-adjacent long-bone surfaces.

If triggered, the better fracture-handling baseline—not automatically the model with the highest pooled Dice—should be extended. Any new architecture must form a new versioned experiment and must not be presented as part of the original baseline comparison.

## 8. Recommended thesis results wording at the present cut-off

> A certified cohort of 71 knees from 43 subjects was established, comprising 58 healthy VSD knees and 13 fractured Ruikar knees. All inputs were standardised to LPS orientation and a fixed 200 mm field of view, with four per-bone occupancy targets on a 256³ grid. The Stage 1 gate passed with zero detected split leakage, 284/284 automated sample–bone retention checks passing, 13/13 fracture-region records approved and 142/142 AP/LAT DRRs passing numerical QA.
>
> Five fold-specific ConvNeXtV2 encoder and shared-front-end pathways were trained without test-subject access. All 20 revised decoder runs completed successfully while preserving their frozen front-end states. Across folds, the residual-ReLU and residual-PReLU arms achieved mean best-validation macro Dice values of 0.4635 and 0.4552, compared with 0.4174 and 0.4143 for the plain-ReLU and plain-PReLU arms. These values describe checkpoint selection and must not be interpreted as final test performance.
>
> A complete held-out architecture comparison could not be regenerated at the evidence cut-off because only one of 20 expected OOF metric files was available locally. Accordingly, no architecture was declared superior, no statistical hypothesis tests were performed, and the fracture-aware extension remained inactive pending complete per-subject, per-bone and healthy/fractured evaluation.

## 9. Evidence map

| Claim area | Primary repository evidence |
|---|---|
| Data contract | `configs/data_contract_v1.json` |
| Architecture/training contract | `configs/baseline_protocol_v1.json` |
| Certified cohort and folds | `reports/manifests/quantitative_manifest_v1.csv` and `.metadata.json` |
| Leakage | `reports/manifests/leakage_report_v1.json` |
| Stage 1 independent verdict | `reports/agent_runs/stage1/s1_manifest_certification_v1/independent_gate_review.md` |
| Per-bone retention | `reports/agent_runs/stage1/s1_gt_per_bone_retention_v1/v2_gate_review.md` |
| DRR independent verdict | `reports/agent_runs/stage1/s1_drr_generation_evidence_v1/independent_gate_review.md` |
| P1/P2 histories | `models/foundation_stage2_v1/fcmae_p1_history_fold*.csv`, `fcmae_p2_history_fold*.csv` |
| P1/P2 provenance and pairing | `models/foundation_stage2_v1/fold*/fcmae_*_provenance.json`, `fcmae_p2_pairing_audit_v2.json` |
| Shared front ends | `models/foundation_stage2_v1/fold*/shared_frontend.pth` |
| Capacity gate | `models/foundation_stage2_v1/fold0/03a_results/foundation_summary.json` |
| Combined Stage 2 verdict | `models/foundation_stage2_v1/independent_stage2_gate_verdict.json` |
| Stage 3 summaries | `models/decoders/foundation_stage2_v1/fold_*/**/run_summary.json` |
| Stage 4 implementation | `notebooks/modeling/04_decoder_comparison.ipynb` |
| Metric tests | `notebooks/modeling/06_metric_foundation.ipynb`, `notebooks/modeling/tests/metric_foundation.ipynb` |

## References

Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *Proceedings of Machine Learning Research, 119*, 1597–1607. https://proceedings.mlr.press/v119/chen20j

Gopalakrishnan, V., & Golland, P. (2022). Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging. *Clinical Image-Based Procedures*. https://arxiv.org/abs/2208.12737

Maier-Hein, L., Reinke, A., Godau, P., et al. (2024). Metrics Reloaded: Recommendations for image analysis validation. *Nature Methods, 21*, 195–212. https://doi.org/10.1038/s41592-023-02151-z

Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. https://arxiv.org/abs/1606.04797

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*, 234–241. https://lmbweb.informatik.uni-freiburg.de/Publications/2015/RFB15a/

Shakya, M., & Khanal, B. (2023). Benchmarking encoder-decoder architectures for biplanar X-ray to 3D bone shape reconstruction. *Advances in Neural Information Processing Systems, 36*. https://papers.neurips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html

Woo, S., Debnath, S., Hu, R., et al. (2023). ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. *CVPR 2023*, 16133–16142. https://openaccess.thecvf.com/content/CVPR2023/html/Woo_ConvNeXt_V2_Co-Designing_and_Scaling_ConvNets_With_Masked_Autoencoders_CVPR_2023_paper.html

Wu, Y., & He, K. (2018). Group normalization. *ECCV 2018*, 3–19. https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html
