# Comparison Architecture and Findings — 3D U-Net vs V-Net Knee Reconstruction

**Project:** Fracture-Preserving 3D Knee Reconstruction from Bi-planar X-rays — A Controlled Comparison of U-Net-style and V-Net-style Decoder Blocks
**Prepared:** 2026-07-22
**Scope:** Answers five specific questions — comparison method, training architecture, rationale, data-exploration yield, and pre-processing — each grounded in files in this repository and supported by the academic sources actually used to justify the design choices. It complements, and does not replace, [`docs/stage1_stage2_writeup.md`](stage1_stage2_writeup.md); where the two disagree on a number, this document is the more current one (see §6 for why).

> **Read this before the rest.** No accepted U-Net-vs-V-Net accuracy numbers exist yet. The Stage 2 combined gate was passed on 2026-07-21 (`models/foundation_stage2_v1/independent_stage2_gate_verdict.json`, `status: "PASS"`), which *unblocks* full training — it is not a result. Two of four arms exist as pre-amendment pilot runs for folds 1–2 only (`HPC/HPC_results/decoder_result/fold_1/`, `fold_2/`), and the gate verdict explicitly forbids reusing them (`"do_not_reuse_pre_amendment_unweighted_decoder_runs"`) because the training split and loss weighting changed after they ran. All 20 primary-analysis runs (5 folds × 4 arms) plus an 8-run seed sweep still need to execute on the revised protocol. What follows is the architecture and the methodological findings produced while building and QA-gating the apparatus — not decoder-comparison results.

---

## 1. How is the comparison done?

The comparison is a **matched 2×2 factorial ablation of the decoder block**, conditional on an identical, frozen, non-decoder front end, evaluated by 5-fold subject-grouped cross-validation with a pre-registered effect-size decision rule.

### 1.1 The four arms

The original plan compared two named baselines — a "U-Net-style" block and a "V-Net-style" block — but these two differ on **two design axes at once**: skip topology (plain vs. residual) and activation function (ReLU vs. PReLU). A Dice or ASSD gap between them cannot be attributed to either axis alone, so the design was expanded to a full 2×2 factorial that fills in the two missing corners (`reports/agent_runs/stage2/s2_3_decoder_2x2_factorial/session_summary.md`, §1–2; `decoder_cv_protocol_amendment_v1.md`, line 12):

| Arm | Activation | Residual skip | Corresponds to |
|---|---|---|---|
| `plain_unet_style` | ReLU | No | original "U" |
| `residual_vnet_style` | PReLU | Yes | original "V" |
| `residual_relu_style` | ReLU | Yes | added corner |
| `plain_prelu_style` | PReLU | No | added corner |

All four arms are built from one parameterized `DecoderBlock(activation, residual)` rather than two hand-written classes, share identical initialization (hash-recorded), and are proven numerically identical to the original hand-written blocks via `torch.allclose` (`block_equivalence_test`, `session_summary.md` §2). Everything else — channel schedule, skip concatenation, upsampling, loss, optimizer, augmentation, seed, and stopping rule — is held fixed across arms (`notebooks/modeling/03b_decoder_cross_validation.ipynb`, cell 0: *"deliberately matched adaptations, not faithful reproductions of the complete original 3D U-Net and V-Net"*).

### 1.2 Cross-validation design

- **5-fold, subject-grouped** split (`StratifiedGroupKFold` on `subject_id`), seed 42, validation fold = `(test_fold + 1) mod 5` (`configs/baseline_protocol_v1.json`).
- Primary cohort: **71 knees, 43 subjects** (all 5 test folds). A **pre-registered sensitivity analysis** drops `test_fold == 4` (expected 57 knees / 34 subjects) because that fold's shared front end shows a diagnosed representation defect (§3.2).
- Run budget: **20 primary runs** (5 folds × 4 arms) + **8 seed-sweep runs** (fold 0 only, seeds 123/2024, reused 42) to give a training-variance band to judge any observed decoder effect against.
- Leakage controls, all verified zero: no subject spans two test folds, no rows in the forbidden fold 5, no train/val/test overlap, no augmentation child crossing its parent's fold (`reports/manifests/leakage_report_v1.json`).

### 1.3 Endpoints and statistical test

- **Co-primary endpoints**: subject-level macro Dice and subject-level macro ASSD (mm). Bilateral knees are averaged within-subject *before* the group comparison, so the statistical unit is the subject, not the knee — avoiding pseudoreplication from double-counting two knees per person.
- **Co-primary contrasts**: the *residual main effect* and the *activation main effect*, each summarized on both endpoints (4 contrast-endpoint pairs total) as a descriptive effect size, with no significance test or multiple-comparison correction applied.
- **Effect size**: raw and favourable-direction (per-subject) mean and median differences for each contrast, reported without a hypothesis test.
- **Confidence intervals**: 10,000-resample, fold-stratified subject bootstrap for each paired effect.
- **Decision rule**: a factor (residual or activation) is declared "preferred" only if the median favourable effect agrees in direction on both endpoints; otherwise the result is `INCONCLUSIVE` — a descriptive, effect-size-only rule with no significance test, chosen because prior evidence (Isensee & Maier-Hein, 2019; Isensee et al., 2024) shows residual-vs-plain decoder effects in 3D U-Nets are typically small, so a null or ambiguous descriptive result is a realistic, honest outcome rather than a failure.
- The original two-arm "V minus U" contrast is retained only as a **secondary, descriptive** result, since it still confounds two factors.
- Fractured-only subgroup results are reported **descriptively**, not as a formal contrast, because only 13 fractured knees exist.

### 1.4 Pre-registration discipline

`decoder_cv_protocol_amendment_v1.md` is explicitly "frozen before any fold-0, fold-3, or fold-4 decoder result is inspected" — a genuine pre-registration: the fold dispositions, sensitivity analysis, and decision rule were all fixed *before* looking at those folds' results, which is what makes the eventual `INCONCLUSIVE`/`ROBUST`/`FOLD4_DEPENDENT` labels credible rather than post-hoc.

---

## 2. What is the training architecture?

```
ConvNeXtV2 encoder (self-supervised, per-fold)
        │  FCMAE masked-patch pretraining (P1)
        │  bidirectional AP↔LAT cross-view completion (P2)
        ▼
Shared bi-planar fusion + 3D lift ("front end")
        │  trained once per fold through a disposable neutral head,
        │  head discarded, remainder frozen and exported
        ▼
Decoder (one of 4 factorial arms — the only thing that changes)
        │  U-Net-style / V-Net-style / and 2 crossed variants
        ▼
[B, 4, 256, 256, 256] logits → femur, tibia, patella, fibula occupancy
```

### 2.1 Encoder (Stage 2.1, `notebooks/modeling/01_encoder_pipeline.ipynb`)

- **ConvNeXtV2** (Woo et al., 2023), initialized from the pinned public checkpoint `convnextv2_tiny.fcmae` (not random init — its origin, URL, license, and state hash are recorded).
- **P1 — FCMAE pretraining**: reconstructs uniformly masked DRR patches, following the fully convolutional masked-autoencoder recipe ConvNeXtV2 was co-designed with (Woo et al., 2023). 128 uniform draws with 1 optimizer update per epoch; smoke-tested at a 32-pixel patch contract.
- **P2 — cross-view completion**: adds bidirectional AP↔LAT reconstruction on top of the retained FCMAE loss, teaching the encoder to reason jointly across the two orthogonal projections.
- Five encoders are trained, one per fold, on that fold's training subjects only; test-fold files are opened only for provenance bookkeeping, never for gradients. Augmentation is photometric-only (gamma, brightness, Gaussian noise) and deterministically seeded; no geometric augmentation is used anywhere in the pipeline, since it would corrupt the bi-planar projection geometry the 3D lift depends on.

### 2.2 Encoder feature audit (Stage 2.1b, `01b_encoder_feature_audit.ipynb`)

An evaluation-only stage (no weight updates) that checks the encoder actually improved rather than regressed, using two tracks: local four-bone linear probes on L0/L1 features, and global AP↔LAT retrieval (recall, MRR, permutation test, effective-rank collapse checks) on pooled L2/L3 features. This audit is what later surfaced the fold-4 representation defect described in §3.2.

### 2.3 Shared front end (Stage 2.2, `notebooks/modeling/02_frontend_pretrain.ipynb`)

- Loads the fold's approved P2 encoder and trains the **complete encoder + bi-planar fusion + orthogonal 3D lift** end-to-end through a disposable neutral head (deep-supervision style, 4 multiscale 1×1×1 heads), for 40 epochs, then **discards the head** and exports a frozen, head-free `shared_frontend.pth`.
- **Scale-dependent fusion**: local convolutional fusion at 64×64 and 32×32 (where spatial detail matters and attention is unaffordable); learned cross-attention at 16×16 and 8×8 (where global AP/LAT context is useful and the token count is manageable). Fusion is bidirectional — AP is refined by LAT and vice versa.
- **Geometry-locked orthogonal lift**: each fused 2D feature map is expanded along its own projection axis (AP → one world axis, LAT, flipped → the orthogonal axis) into a common LPS-aligned 3D grid — a fixed, non-learned orientation step, not a learned deprojection.
- **Output contract**: feature channels `[64, 128, 256, 512]` at spatial resolutions `[64³, 32³, 16³, 8³]`.
- Four independent, ordered supervised targets `[femur, tibia, patella, fibula]` — never a single binary-union or TSDF target (both explicitly forbidden in `configs/baseline_protocol_v1.json`).
- The export is verified deterministic: two forward passes on the same fixed input are checked byte-identical by SHA-256.

### 2.4 Decoders (Stage 2.3, `03_decoder_pipeline.ipynb` / `03a_decoder_capacity_gate.ipynb` / `03b_decoder_cross_validation.ipynb`)

- Both decoder styles share one `Decoder3D` skeleton: three transposed-convolution up-steps with skip concatenation from the front end's feature pyramid, then trilinear refinement, then a 1×1×1 conv to 4 output channels. Only the block type (`DecoderBlock(activation, residual)`) differs between arms.
- **Normalization**: GroupNorm, 8 groups, everywhere — no BatchNorm — justified for batch-size-1 3D training (Wu & He, 2018).
- **Loss**: `0.5 · BCE + 0.5 · soft-Dice`, with per-bone BCE positive weights computed only from the current fold's training targets.
- **Resolution ceiling**: both decoders emit logits at 128³, then a shared trilinear upsample (`align_corners=False`) to the 256³ target grid — a deliberate compute/memory tradeoff in the style of decoder-cost-reduction work such as EffiDec3D (Rahman & Marculescu, 2025), but it means the reported 0.78125 mm ASSD outruns the model's true resolvable precision (effective spacing 1.5625 mm); every run now machine-records `logit_resolution=128` so this ceiling is explicit rather than implied.
- **Enforced fairness contract**: the notebook refuses to run unless both arms have zero BatchNorm layers, exactly 12 GroupNorm layers each, and identical wiring points (`up3/up2/up1/refine128/192/256/output`).
- **Capacity ("overfit") gate**: on 1- and 2-case subsets, each arm must reach ≥ 0.90 hard Dice on every bone before full cross-validation unlocks — proof the decoder has enough capacity to fit the target before spending a 5-fold budget on it. Two arms failed this gate on `subset_1_healthy` in the most recent run; the project owner explicitly authorized proceeding to final training anyway (`models/foundation_stage2_v1/independent_stage2_gate_verdict.json`, `capacity_disposition.status: "ACCEPT_WITH_PROJECT_OWNER_OVERRIDE"`) while preserving the failure in provenance rather than relabeling it a pass.

### 2.5 Metrics core (`notebooks/modeling/06_metric_foundation.ipynb`)

A single canonical metric implementation, reused byte-identically across the decoder, comparison, and test notebooks:
- **Overlap**: Dice, IoU — with explicit invalid/empty handling (empty target → NaN, flagged invalid; empty prediction against a valid target → Dice 0, never silently dropped, so a missing bone cannot hide inside a macro average).
- **Surface distance**: HD95 and ASSD (mm), computed from eroded surface masks with spacing-aware Euclidean distance transforms (anisotropic voxel spacing respected).
- **Fracture-aware diagnostics**: connected-component agreement, a `false_bridge` flag when the model "heals" a fracture gap the target preserves, and minimum component-gap in mm — built specifically to catch a decoder that closes a fracture line it should have kept open.
- Subject-first aggregation (bilateral knees averaged before the cohort mean).
- All 25 synthetic analytic checks pass.

---

## 3. Why is the comparison done in this method?

### 3.1 Isolating the decoder as the only variable

The central methodological claim is narrow and explicit: *"The comparison must isolate decoder design. A single fold-specific encoder/fusion/lift is therefore trained once, frozen, and supplied unchanged to both decoder arms. If either decoder fine-tuned this front end, performance differences could no longer be attributed to the decoder alone."* (`02_frontend_pretrain.ipynb`, cell 0). Fine-tuning the front end during decoder training is explicitly forbidden as the primary-comparison regime (`configs/baseline_protocol_v1.json`, `forbidden.regimes`). This mirrors the standard ablation logic used to attribute performance differences to a single architectural factor, as in the controlled decoder-choice comparisons underlying UNETR (Hatamizadeh et al., 2022) and the broader caution — reinforced by nnU-Net's own authors — that architectural claims must be validated against a properly matched baseline rather than a differently-tuned one (Isensee et al., 2024).

### 3.2 Why the design was expanded from 2 arms to a 2×2 factorial

The original two-arm comparison confounded skip topology and activation function. Splitting these into an orthogonal 2×2 design lets the analysis attribute an observed effect to a residual main effect, an activation main effect, or their interaction, rather than to an unresolvable bundle of both — the same logic that makes any controlled ablation informative rather than merely descriptive.

### 3.3 Why `INCONCLUSIVE` is pre-declared as a legitimate outcome

With **n = 43 subjects** and a single training seed per arm, prior evidence — Isensee & Maier-Hein (2019) and the more recent *nnU-Net Revisited* validation study (Isensee et al., 2024) — shows that residual-vs-plain decoder effects in 3D U-Nets are typically small. The protocol therefore pre-declares that a null or ambiguous result is a realistic, honest outcome and encodes this directly into the decision rule (directional agreement of the median favourable effect across both co-primary endpoints, else `INCONCLUSIVE`) rather than treating "no clear winner" as a failed experiment.

### 3.4 Why Dice is paired with ASSD, not reported alone

*Metrics Reloaded* (Maier-Hein et al., 2024) recommends problem-aware metric selection, including boundary-sensitive evidence whenever boundary accuracy is part of the domain interest (as it is for fracture lines and thin bones like the fibula) and explicit handling of empty predictions — both of which motivate pairing an overlap metric (Dice) with a surface-distance metric (ASSD) as co-primary endpoints, rather than relying on Dice in isolation.

### 3.5 Why healthy/fractured subgroups are reported disaggregated

Reporting healthy and fractured subgroup performance separately, rather than only a pooled average, follows the disaggregated-reporting recommendation from the closest directly comparable benchmark in this exact problem space — a systematic benchmark of encoder-decoder architectures for bi-planar X-ray to 3D bone shape reconstruction (Shakya & Khanal, 2023) — because a pooled macro Dice can hide a decoder that trades fracture-case accuracy for healthy-case accuracy or vice versa, which is precisely the axis this capstone's conditional stretch goal (fracture-aware architecture) cares about.

### 3.6 Why GroupNorm instead of BatchNorm

3D volumes at 256³ force a small (effectively 1) batch size under memory constraints. Batch Normalization's statistics become unreliable at small batch sizes; Group Normalization is batch-size-independent and was shown to substantially outperform BatchNorm in exactly this regime (Wu & He, 2018), which is why it is enforced as a hard architecture-contract requirement rather than a preference.

### 3.7 Why the QA gate is independent and pre-registered, not self-certified

The repository's own contract (`CLAUDE.md`) blocks full cross-validation until an *independent* Stage 2 QA gate passes — a deliberate check against the experimenter grading their own homework. When the automated feature audit produced ambiguous `WARN` verdicts for three of five folds, the response was a formal, hash-anchored amendment frozen *before* those folds' decoder results were inspected (§1.4), rather than an ad hoc adjustment made after seeing favorable or unfavorable numbers — the standard justification for pre-registration in small-sample comparative studies.

---

## 4. What is the data-exploration yield?

### 4.1 Final certified cohort

| Dataset | Fracture status | Ready | Excluded |
|---|---|---|---|
| VSD (healthy) | healthy | **58** | 4 |
| Ruikar (fractured) | fractured | **13** | 3 |
| **Total (quantitative cohort)** | | **71** | **7** |
| Regen (clinical X-rays, illustrative only) | — | 30 folders / 26 unique patients | not applicable |

Source: `reports/manifests/quantitative_manifest_v1.csv` and `.metadata.json` — `included_planning_rows: 71`, `ready_rows: 71`, `pending_recertification_rows: 0`, `excluded_rows: 7`, `certification_approved: true`, cohort contract `"58 healthy, 13 fractured"` satisfied.

### 4.2 How the 71-knee cohort was reached

- **Regular VSD** (lower-limb DICOM): started from a downloaded set of 6 subjects; 1 (VSD010, initially thought too short in z-extent) was later re-certified after its split-folder DICOM was correctly merged (`vsd_multi_merge.ipynb`) rather than excluded — the earlier apparent short z-extent was a filename-ordering artifact, not a real coverage gap. Final: **11 subjects → 22 knee volumes** (8 single-folder + 3 merged subjects, ×2 legs).
- **Z-prefix VSD** (full-body CT, discovered later in exploration): **20 subjects**, of which `z057` (both sides) was excluded for noise/metal, and `z050`/`z063` each had one side excluded for total-knee-replacement metal artifact — contributing **36 usable knee volumes**.
- **Ruikar (fractured)**: **16 source cases** (4 PartLeft + 12 PartRight); 2 excluded as single-slice scout series (`Case4`, `Case10`, below the minimum Z-slice threshold), 1 further excluded for fracture-fixation metal hardware (`Case8`) — leaving **13 usable fractured knees**. Two additional cases (`Case3`, `Case16`) needed manual cast-cleanup rather than exclusion (§5).
- **22 + 36 = 58 healthy** + **13 fractured = 71**, matching the certified manifest exactly (`notebooks/pre-processing/predrr_preprocessing.ipynb`, cell 0 dataset table).
- **Zero leakage** on every checked axis: no subject spans two test folds, no rows in the forbidden fold 5, no augmentation-parent mismatch, no train/val/test overlap for any fold (`reports/manifests/leakage_report_v1.json`).

### 4.3 Regen (clinical X-rays) — kept separate

30 patient folders, 60 DICOM images, 90 annotation files; 4 patients have repeat visits, so **26 unique patients**. This dataset has no paired CT ground truth and is explicitly excluded from quantitative decoder selection — used only for qualitative, clinician-reviewed visual assessment (`04_decoder_comparison.ipynb`, success criterion: *"Regen outputs... are labelled illustrative; they cannot select the winning decoder"*).

---

## 5. How is the data pre-processed?

### 5.1 CT standardization (`notebooks/pre-processing/predrr_preprocessing.ipynb`)

Applied uniformly to all 71 volumes:

```
Load → Resample (0.5 mm isotropic intermediate) → Orient to LPS → Bone window [-450, 1050] HU
     → Body-envelope mask (drop CT table + stray body parts) → ROI bone crop
     → Center into a fixed 200 mm physical FOV cube → Resize to 256³ → save NIfTI, float [0,1]
```

- **Orientation**: the authoritative convention is **LPS**, end-to-end; the pipeline's own "Bugs Fixed" list records that an earlier LAS orientation bug was corrected.
- **Fixed-FOV rationale (a genuine methodological finding)**: an earlier `pad_to_cube` approach sized each output cube to the longest crop axis per volume. Because fractured scans include extra femur/tibia shaft length (213–309 mm Z) relative to the healthy ±100 mm knee crop (~199 mm Z), this would have given fractured knees a coarser effective voxel spacing (~0.83–1.21 mm vs. ~0.78 mm) and different bone occupancy — a data-preprocessing confound that would bias any healthy-vs-fractured comparison before the decoder ever saw the data. Fixing the FOV to a constant 200 mm cube for every volume makes spacing, scale, and bone occupancy constant across the healthy and fractured cohorts, protecting the downstream decoder comparison from this confound.
- **Body-envelope mask**: keeps the largest soft-tissue connected component and fills holes, automatically removing the CT table/support slab and disconnected extra body parts for every case, including air-gap-separated table cases.

### 5.2 Artifact cleanup

- **Fused cast (Case3, Case16)**: a dense cylindrical cast (~200 HU) is topologically fused to the limb and cannot be separated by thresholding or connected-component analysis alone. Removed manually in 3D Slicer first, then automated speck-cleanup (`notebooks/pre-processing/fractured_cast_cleanup.ipynb`) keeps the main bone plus any dense fragment within 15 mm (spacing-aware), dropping the rest.
- **Contralateral-leg contamination (VSD z036)**: the bilateral field of view captured opposite-leg bone that a single shared soft-tissue envelope cannot separate; resolved by erosion-based leg separation (`notebooks/pre-processing/z036_contralateral_cleanup.ipynb`).
- **Metal/TKR exclusion**: `z050`/`z063` (one side each) and fractured `Case8` are excluded outright rather than cleaned, since metal saturates the HU ceiling and occludes true bone shape — judged unrecoverable rather than a cleanup target.
- **Orientation correction**: a connected-component "blob heuristic" flags superior–inferior flips (femur = 1 blob at one end; tibia + fibula = 2 blobs at the other); a separate notebook (`ground_truth_orientation_fix.ipynb`) performs a header-only sign correction for ground-truth voxelization, since a naive image-flip would be a physical no-op for a header-sign bug.
- **Multi-folder DICOM merge (VSD 010, 015, 017)**: split lower/upper-leg folders merged by physical `ImagePositionPatient[2]` position (not filename order, which was proven to undercount z-coverage), gap tolerance ≤ 5 mm, minimum span ≥ 700 mm.

### 5.3 DRR generation (`notebooks/pre-processing/drr_generation.ipynb`, DiffDRR)

- Renders **one AP and one LAT view per knee** using **DiffDRR**, an auto-differentiable, GPU-accelerated digitally-reconstructed-radiograph renderer built on a vectorized Siddon's-method ray-tracer in PyTorch (Gopalakrishnan & Golland, 2022) — chosen so the DRR generator is differentiable and GPU-batchable rather than a fixed offline CPU tool.
- Geometry: SDD (source-detector distance) = 1000 mm, SOD (source-object distance) = 850 mm (~1.18× magnification), detector spacing 1.4 mm, output 256×256 to match the volume grid.
- 71 volumes → **142 DRRs**, saved as float32 `.npy` (model input) plus 16-bit PNG (QA).
- **Density subtlety**: DiffDRR's internal Beer–Lambert attenuation model classifies air/soft-tissue/bone by fixed HU thresholds, so the [0,1]-normalized volumes must first be reversed back to pseudo-HU before rendering — otherwise every voxel reads as soft tissue and the bone-attenuation multiplier has no effect. This was caught and validated on a CPU smoke test before the full GPU batch.

### 5.4 Ground truth (`notebooks/modeling/00_gt_per_bone.ipynb`)

Manually segmented Slicer STLs (femur, tibia, patella, fibula) are voxelized onto each case's pre-processed grid, producing the 4-channel binary occupancy target every decoder arm reconstructs; regular-VSD STLs are segmented from the raw/healthy source, z-prefix STLs from the external/upright source, selected per case.

### 5.5 Target resolution

Final stored/target grid: **256³ voxels at 0.78125 mm isotropic** (200 mm FOV ÷ 256), unified across the local Windows environment and the Sunway HPC environment.

---

## 6. Why this document differs from `docs/stage1_stage2_writeup.md`

That document (prepared 2026-07-16) describes a state where the cohort manifest was `certification_approved: false`, the decoder comparison was still a confounded 2-arm design, and every modelling notebook shipped with `RUN_REAL_DATA=False` — i.e., no runs had started. In the six days since, the cohort was certified (`certification_approved: true`), the comparison was redesigned into the 2×2 factorial described in §1, a combined independent Stage 2 QA gate was reached and passed with an owner override (§2.4), and two pilot folds of decoder training ran under the *old*, now-superseded protocol. None of that pilot training data may be reused under the amended protocol (§1 caveat), so this document's factual claims about cohort size, comparison design, and gate status are the current ones; treat `stage1_stage2_writeup.md` as a snapshot of an earlier, still-instructive stage of the same project rather than a contradiction.

---

## References

Gopalakrishnan, V., & Golland, P. (2022). Fast auto-differentiable digitally reconstructed radiographs for solving inverse problems in intraoperative imaging. In *Clinical Image-Based Procedures Workshop (CLIP), in conjunction with MICCAI 2022* (pp. 1–11). Springer. https://doi.org/10.1007/978-3-031-23179-7_1

Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H. R., & Xu, D. (2022). UNETR: Transformers for 3D medical image segmentation. In *2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)* (pp. 1748–1758). IEEE. https://doi.org/10.1109/WACV51458.2022.00181

Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2019). *nnU-Net: Self-adapting framework for U-Net-based medical image segmentation* (arXiv:1908.02182). arXiv. https://arxiv.org/abs/1908.02182

Isensee, F., Wald, T., Ulrich, C., Baumgartner, M., Roy, S., Maier-Hein, K., & Jaeger, P. F. (2024). nnU-Net revisited: A call for rigorous validation in 3D medical image segmentation. In *Medical Image Computing and Computer Assisted Intervention (MICCAI 2024)* (pp. 488–498). Springer. https://doi.org/10.1007/978-3-031-72114-4_47

Kasten, Y., Doktofsky, D., & Kovler, I. (2020). End-to-end convolutional neural network for 3D reconstruction of knee bones from bi-planar X-ray images. In *Machine Learning for Medical Image Reconstruction (MLMIR 2020), in conjunction with MICCAI 2020* (pp. 123–133). Springer. https://doi.org/10.1007/978-3-030-61598-7_12

Lin, Y., Sun, H., Li, Y., et al. (2026). Real-time reconstruction of 3D bone models via very-low-dose protocols. *npj Digital Medicine*. https://www.nature.com/articles/s41746-026-02389-9

Maier-Hein, L., Reinke, A., Godau, P., Tizabi, M. D., Buettner, F., Christodoulou, E., Glocker, B., Isensee, F., Kleesiek, J., Kozubek, M., Reyes, M., Riegler, M. A., Wiesenfarth, M., Kavur, A. E., Sudre, C. H., Baumgartner, M., Eisenmann, M., Heckmann-Nötzel, D., Rädsch, T., … Jäger, P. F. (2024). Metrics reloaded: Recommendations for image analysis validation. *Nature Methods, 21*(2), 195–212. https://doi.org/10.1038/s41592-023-02151-z

Rahman, M. M., & Marculescu, R. (2025). EffiDec3D: An optimized decoder for high-performance and efficient 3D medical image segmentation. In *2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 10435–10444). IEEE. https://openaccess.thecvf.com/content/CVPR2025/html/Rahman_EffiDec3D_An_Optimized_Decoder_for_High-Performance_and_Efficient_3D_Medical_CVPR_2025_paper.html

Shakya, M., & Khanal, B. (2023). Benchmarking encoder-decoder architectures for biplanar X-ray to 3D bone shape reconstruction. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track*. https://proceedings.neurips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html

Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I. S., & Xie, S. (2023). ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders. In *2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 16133–16142). IEEE. https://doi.org/10.1109/CVPR52729.2023.01548

Wu, Y., & He, K. (2018). Group normalization. In *Computer Vision – ECCV 2018* (pp. 3–19). Springer. https://doi.org/10.1007/978-3-030-01261-8_1
