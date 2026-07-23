# Session summary — decoder 2×2 factorial redesign + Stage 2 QA diagnosis

## 1. Plan review: U-style vs V-style decoder ablation

Reviewed `PLAN (new 02-04).md` against `02_frontend_pretrain.ipynb`, `03_decoder_pipeline.ipynb`,
`04_decoder_comparison.ipynb`. Verified every cited paper against academic sources before relying
on it. Findings:

- **Confounded "ablation."** The two original arms (`plain_unet_style` = ReLU/no-residual,
  `residual_vnet_style` = PReLU/residual) differ on **two axes at once** — skip topology and
  activation. A Dice/ASSD gap could not be attributed to either alone. This is not a true ablation.
- **Likely underpowered.** [Isensee & Maier-Hein 2019](https://arxiv.org/abs/1908.02182) and
  [nnU-Net Revisited (MICCAI 2024)](https://arxiv.org/abs/2404.09556) show residual-vs-plain
  effects in 3D U-Nets are typically small; with n=43 subjects and a single seed, `INCONCLUSIVE`
  is a realistic, legitimate outcome that should be pre-declared, not treated as failure.
- **128³→256³ resolution ceiling.** Both decoders emit logits at 128³ then trilinear-upsample to
  256³, so reported ASSD at 0.78125 mm spacing outruns the model's real resolvable precision —
  especially relevant for small bones (patella, fibula) and fracture lines.
- **Well-executed elsewhere:** subject-level aggregation before pairing (avoids bilateral-knee
  pseudoreplication), hash-verified frozen front end, GroupNorm justified for batch-size-1
  ([Wu & He 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Yuxin_Wu_Group_Normalization_ECCV_2018_paper.html)),
  Dice+ASSD co-primary per
  [Metrics Reloaded](https://doi.org/10.1038/s41592-023-02151-z), disaggregated healthy/fractured
  reporting per [Shakya & Khanal, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html).

## 2. Implementation: three requested edits + refactor

User directive: (1) add a 3rd arm to isolate residual vs activation → upgraded to a **full 2×2
factorial** per user's choice; (2) the rest of the plan's suggestions; (3) refactor to remove
unnecessary code. **Scope: HPC notebooks only** (`HPC/HPC_notebooks/modeling_HPC/`); local twins
under `notebooks/modeling/` deliberately left untouched (divergent, older scaffolds).

### 2×2 factorial (was 2 arms → now 4)

Collapsed the two hand-written block classes into one parameterized `DecoderBlock(activation,
residual)` so the four arms differ **only** on the intended axes:

| Arm | Activation | Residual | Label |
|---|---|---|---|
| `plain_unet_style` | ReLU | No | U (original) |
| `residual_vnet_style` | PReLU | Yes | V (original) |
| `residual_relu_style` | ReLU | Yes | new |
| `plain_prelu_style` | PReLU | No | new |

- `make_activation` raises `ValueError` on unknown values — no silent PReLU fallthrough.
- All four arms share identical conv/norm initialisation (`build_matched_decoder`, hash-recorded)
  so random init isn't a hidden confound.
- `block_equivalence_test` proves the unified block is **numerically identical**
  (`torch.allclose`) to the original hand-written blocks — no parameter-count-equality claim made.
- `04` now runs **predefined contrasts**: decoder-family diagonal (V−U, kept for continuity),
  residual main effect, activation main effect, residual×activation interaction, and a descriptive
  pathology (healthy vs fractured) interaction. Co-primary = the two main effects × {Dice, ASSD},
  Holm-corrected across 4; a factor is "preferred" only if both endpoints are significant in the
  same direction **and** the interaction is non-significant, else `INCONCLUSIVE`.

### Training-variance band

Added a fold-0 seed sweep (`RUN_SEED_SWEEP`, seeds 123/2024, reusing 42 from main CV) → 8 extra
runs. `04` loads per-arm across-seed dispersion of subject-macro Dice/ASSD so an observed effect
can be judged against training-run variance, not just against zero.

### Resolution recording

Logits stay at 128³ then trilinear→256³ (no OOM risk — user's explicit choice), but every
config/summary/OOF record now carries `logit_resolution=128` and
`effective_spacing_mm=1.5625` so ASSD's real precision ceiling is machine-recorded, not implied.

### Refactor + bugfixes

- **`02`:** fixed a latent `NameError` — `USE_ACTIVATION_CHECKPOINTING` was referenced in
  `frontend_config` but never defined; set to literal `False` (bridge doesn't use checkpointing).
  Removed a duplicated split-assignment block in `load_certified_folds`.
- **`03`:** removed confirmed-dead code (`per_bone_metrics`, `aggregate_subject_length`,
  `hd95_mm` — HD95 is computed inline elsewhere), unused imports (`matplotlib.pyplot`, `math`),
  unused constants (`PROTOCOL_VERSION`, `FOLD`, `TARGET_SIZE`), duplicate `BONES` redefinition.
- Citations/rationale prose left untouched throughout; only count-bearing sentences the arm
  change made stale were updated ("two arms"→four, "ten runs"→twenty).

### Verification (data-free, executed against actual notebook cell source, not retyped copies)

- `03`: all 4 arms build; GroupNorm validation (`num_groups==8`, `num_channels%8==0`, no
  BatchNorm/InstanceNorm) holds; block equivalence passes; matched-init yields one shared hash
  across arms; guard tests reject bad activations/arm names; all arms forward to
  `[1,4,256,256,256]`.
- `04`: factorial analysis on a synthetic 43×4 table recovers an injected signal; Holm applies to
  exactly the 4 co-primary rows; decision rule fires correctly.
- Config (`configs/baseline_protocol_v1.json` `decoder_conditions`) updated to list all 4 arms.

Run budget: main CV 5 folds × 4 arms = **20 runs**; seed sweep = 8 more. Full execution remains
gated on independent Stage 2 QA PASS (see §4).

## 3. What `02_frontend_pretrain.ipynb` does (explainer)

Stage 2.2. Per fold: loads the fold's frozen P2 encoder (never updated), trains only the
fusion/projection/lift "bridge" through four disposable multiscale 1×1×1 heads (deep-supervision
style, so no trainable 256³ decoder contaminates the comparison), then discards the heads and
exports a frozen, head-free `shared_frontend.pth`. Guarantees enforced at export: encoder hash
unchanged before/after, no supervision-head parameters leak into the export, reloaded front end is
deterministic with exact expected feature shapes, GPU memory stays under 90%. This shared front end
is what all four decoder arms in `03` load and freeze — the scientific control that isolates the
decoder comparison from representation differences.

## 4. Stage 2 QA gate: diagnosis and resolution path

### The error and why it's not a bug

`require_independent_qa()` in `03` raised `fold {N} independent Stage 2 QA PASS is required;
current quality=WARN` — this is `CLAUDE.md`'s own contract ("Full U-Base/V-Base cross-validation
is blocked until the Stage 2 independent QA gate passes") working as intended, not a defect.

### Root cause (before `pinned_init` upload)

All 5 folds' P2 audits were `quality_status=WARN` for two reasons:
1. `reference_comparison_incomplete` — the `pinned_init` (off-the-shelf `convnextv2_tiny.fcmae`
   checkpoint) comparison stage had never been run for any fold.
2. `p2_preservation_warning` — P2 barely changed retrieval/local metrics vs P1 on several folds.

**Correction made mid-session:** `pinned_init` is the public pretrained checkpoint, not a
random-init baseline as first assumed from stale memory — corrected before drawing conclusions.

### After user uploaded `pinned_init` + re-ran P2 for all 5 folds

`reference_comparison_incomplete` cleared everywhere (`stage_comparison: COMPLETE`). Result:

| Fold | `quality_status` | Remaining warning(s) |
|---|---|---|
| 0 | WARN | `p2_preservation_warning` |
| 1 | **PASS** | — |
| 2 | **PASS** | — |
| 3 | WARN | `p2_preservation_warning` |
| 4 | WARN | `global_feature_evidence_below_threshold`, `p2_preservation_warning` |

### Deep-dive diagnosis (folds 0, 3, 4)

Pulled full global-retrieval metrics, L2/L3 feature-spectrum statistics, P2 training histories, and
pairing audits across all three stages (`pinned_init`/P1/P2) and all 5 folds.

- **Folds 0 and 3 — sub-noise fluctuation, not real regression.** `p2_preservation` fails on a
  bare `mrr_change > 0` with zero tolerance (unlike the local check's ±0.02 band). Fold 0's
  `Δmrr=−0.0071` traces to exactly one query slipping one rank (recall@5 actually *improved*
  0.893→0.929, permutation p unchanged at 0.0010). Fold 3's `Δmrr=−0.0036` came with recall@1
  *improving* (0.179→0.214) and the paired-cosine-margin CI floor more than 5× stronger
  (0.00036→0.00193); both folds report `global: PASS` and beat the public checkpoint by a wide
  margin. Caveat: fold 3's L2 effective-rank did genuinely collapse (5.73→1.25), though L3 stayed
  healthy and carried the descriptor.
- **Fold 4 — genuine dimensional collapse, categorically different.** L3 effective rank fell
  8.0→2.17 with 84% of variance in one direction (only fold showing collapse at *both* L2 and
  L3). Consequences: recall@1=0.067 (exact chance for 15 pairs), permutation p=0.55 (null), and a
  **negative** paired-cosine-margin (−0.00282) — true AP/LAT pairs less similar than mismatched
  ones. Corroborated independently by the pairing audit: fold 4's median subject margin is
  50–77× smaller than every other fold, only 5/9 subjects show positive margin (coin flip).
  Critically, **no training-loss signal caught this** — fold 4 had the *best* validation loss of
  all five folds; the audit's hard gate only checks feature magnitude (`aggregate_std`), not
  spectral rank, so collapse sails through undetected until the soft-WARN stage.

### Decision reached: proceed, with a pre-registered sensitivity analysis (not a blanket caveat)

Conclusion: proceeding to decoder training is defensible because the shared front end is a
constant across all 4 arms, so encoder weakness doesn't bias the *decoder contrast* even though it
lowers everyone's ceiling. However, "state it as a limitation" only correctly covers folds 0/3
(noise) and the general power concern — **fold 4 is not a limitation, it's a broken
measurement** (negative pairing margin = anti-correlated with subject identity) and folding it
into a single vague caveat would misrepresent it.

**Recommended path (not yet implemented):**
- Run all 5 folds as the pre-registered primary analysis (71 knees, 43 subjects).
- Add a **pre-registered fold-4-excluded sensitivity analysis** (56 knees, ~34 subjects) in `04`,
  declared *before* looking at decoder results — if conclusions hold across both, that's stronger
  evidence than either alone; if they diverge, that is itself a reportable finding.
- Optionally, cheapest single lever: re-run P2 for fold 4 with a different seed first (dimensional
  collapse is often init-dependent) — one training run vs. a permanent asterisk on 20% of the
  cohort. Must pre-declare the acceptance criterion before looking, to avoid outcome-driven
  reroll-and-keep selection.
- Do not report macro Dice alone — report the co-primary contrasts (Dice + ASSD on the two main
  effects), and quantify the front-end limitation with the actual numbers above rather than a
  generic statement.

**Not yet done:** wiring the fold-4-excluded sensitivity analysis into `04`, and the actual HPC
execution of the 20 (+8 seed-sweep) decoder runs, which remains gated until folds 0/3/4 dispositions
are finalized per the above.
