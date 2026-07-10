# Modelling Pipeline — Step-by-Step Process & Reasoning

This document summarizes the full modelling pipeline built in `notebooks/modeling/`, covering how a
pair of 2D projection images is turned into a 3D reconstruction of a multi-part target object, and how
two decoder architectures are compared in a controlled, statistically defensible way. It captures
**what each stage does and why it was designed that way**, including the bugs found and fixed along
the way.

**Terminology used throughout:** the object being reconstructed is treated as a generic multi-part
target composed of four structural components (two larger, two smaller). Some instances in the dataset
are geometrically **irregular** (displaced or fragmented sub-regions), and some are **standard**
(intact geometry) — this standard/irregular split is a first-class variable throughout the design,
since the irregular case is the harder and more research-relevant one. Inputs are two 2D **projection
images** per instance (a frontal "View A" and a side "View B"); a subset of instances also has a
**reference 3D capture** available, which is the source of both the projection images (synthetically
rendered from it) and the ground-truth 3D target used for training/evaluation.

---

## 0. Pipeline at a glance

```
00_gt_per_bone.ipynb        Build per-structure 3D ground truth (4 structural components)
        │
01_encoder_pipeline.ipynb   View A + View B → shared ConvNeXtV2 encoder → bi-view fusion → 3D features
        │
02_frontend_pretrain.ipynb  Pretrain the shared front-end (encoder+fusion+lift) per CV fold, neutral head
        │
03_decoder_pipeline.ipynb   Attach Decoder-A OR Decoder-B to the frozen front-end, train, evaluate
        │
04_decoder_comparison.ipynb Pool all folds/regimes/models, compute mean±std + paired significance test
        │
05_decoder_ui.ipynb         Interactive viewer to load any checkpoint and inspect reconstructions
```

**Core design constraint:** both decoders must consume *byte-identical* input features, so that any
performance difference is attributable to the decoder block alone, not to incidental differences in
the shared front-end. This constraint shapes almost every downstream design decision described below.

---

## Step 0 — Per-structure ground truth (`00_gt_per_bone.ipynb`)

Manually-produced surface meshes for each of the four structural components are voxelized onto each
instance's reference 3D grid, written out as one binary volume per component, and checked for
alignment before being used as a training target.

**Reasoning:** an earlier version of the pipeline used a single-channel "occupancy" target derived by
intensity-thresholding the reference 3D capture directly. That approach conflated material density
with structural presence — faint structure in lower-quality captures was under-represented. Splitting
the target into four independent per-component binary channels, built from segmented surfaces rather
than intensity, removes that confound and lets each component be scored (and later inspected)
separately.

---

## Step 1 — Encoder pipeline (`01_encoder_pipeline.ipynb`)

**Goal:** produce reusable multi-scale 3D feature volumes from a pair of 2D projection images (View A
= frontal, View B = side), stopping *before* any decoder or supervised training. This is the part
every downstream model shares.

### 1.1 Data indexing
Paired View-A/View-B projection arrays (256×256 `.npy`, float32 `[0,1]`) are indexed from metadata
CSVs rather than directory-walking, covering both a "standard-capture" and an "augmented" pool. Splits
are done at the **instance level** (not image level) and stratified across the standard/irregular
split, so no instance's data leaks across train/validation.

*Why instance-level:* if two images from the same instance landed in different splits, the model could
"cheat" by memorizing instance-specific appearance rather than learning general projection-to-3D
mapping.

### 1.2 Structure-focused input mask (safety-gated)
Early feature visualizations showed the encoder firing strongly on the outer silhouette of the
projection (the object's outline against its background) rather than on the structural component
itself. A masking step suppresses everything outside a feathered structure-plus-margin region, forcing
the encoder to attend to the actually-relevant content.

**Reasoning + safety check:** because irregular instances can include discontinuous sub-regions, a
naive mask (e.g., "keep the largest connected blob") risks silently deleting a real displaced
sub-region that isn't touching the main structure. Before wiring the mask into the data loader, every
irregular-instance view is passed through a **gate**: the *suppressed* residual (everything the mask
removes) must contain zero structure-intensity pixels. If any instance fails, the mask is not safe to
use. The mask ships with no minimum-blob-size filter and no distance limit for this reason — it only
removes background, never a disconnected structural sub-region.

### 1.3 Self-supervised pretraining (SimCLR)
The backbone (ConvNeXtV2) is pretrained with a contrastive objective (SimCLR / NT-Xent loss) on
**all** available projection images before any supervised step.

**Reasoning:** SimCLR needs no labels, so it can use every image in the dataset without any risk of
leaking test-set *labels* into training. It is pretrained once, globally, and reused across every
cross-validation fold — the only residual risk is that the backbone has *seen* (not learned labels
from) held-out images during this self-supervised phase, which is a standard, accepted trade-off in
low-data settings. A bug in the reference implementation's contrastive loss (the positive-pair labels
pointed at the wrong index) was identified and corrected here.

### 1.4 Hybrid bi-view fusion
The two projections are merged per feature scale with **two different fusion mechanisms**:
- **Fine scales:** a lightweight convolutional mix (concatenate + 3×3 conv + residual) — cheap, and
  preserves high-resolution local detail (important for irregular/fragmented sub-regions).
- **Coarse scales:** full cross-attention between the two views — captures global shape
  correspondence between the two viewpoints, and is affordable because the coarse feature maps have
  few spatial locations.

**Reasoning:** running full attention at the finest resolution would require a several-thousand-token
attention matrix per sample (tens of megabytes just for one map), which is both slow and unnecessary —
local convolution is enough to fuse fine detail, while attention is reserved for where its global
receptive field actually matters.

### 1.5 2D → 3D lift (the critical fix)
Each fused 2D feature map must become a 3D feature volume. The reference implementation "extruded" a
single fused map uniformly along the unobserved (depth) axis — meaning every voxel at a given
depth-slice was identical, so the 3D feature carried **zero information along that axis**. This capped
reconstruction overlap scores at a low ceiling regardless of decoder choice, because the model
architecturally could not encode depth-varying structure.

**Fix:** an **orthogonal back-projection lift**. Each of the two views is placed on the two spatial
axes it actually resolves, and broadcast only along the one axis it cannot see (each view "shadows"
through the volume along its own projection direction, the way two projections of an object intersect
in space). The two resulting volumes are then fused with a 3D convolution. The output volume is a true
cube that varies along all three axes (left-right, front-back, top-bottom).

**Verification:** two guard checks were written specifically to catch a regression back to the old
extruding behavior:
- **Per-axis feature variance check** — asserts standard deviation along all three spatial axes
  exceeds a small threshold; a constant axis (the old bug) fails loudly.
- **Axis-alignment / hull-recall check** — back-projects both raw inputs through the same geometry
  used by the lift and confirms their intersection actually contains the target structure, catching
  any wrong axis or flip convention.

Both checks are wired as `assert` statements, not just printouts, so a future code change cannot
silently reintroduce the bug.

### 1.6 Smoke test
A forward pass on one representative instance from each cohort (standard and irregular) confirms
output shapes match the decoder's expected contract (four feature levels with channel widths and cube
sizes matching the resolution schedule), and the variance/orientation checks both pass before moving
downstream.

---

## Step 2 — Front-end pretraining (`02_frontend_pretrain.ipynb`)

**Goal:** train the fusion + 2D→3D lift (the encoder backbone is kept frozen after step 1's SimCLR
pretraining) against the actual 3D target, using a throwaway decoder head, so that when the real
decoders are attached later, they receive features that already carry useful 3D structure — without
those features having been shaped to favor either decoder.

### 2.1 The "neutral head" trick
Both candidate decoder blocks (see step 3) are built from a shared skeleton; the only difference is
their internal conv block. A **third, minimal block type** — a single conv+norm+activation — is
defined as the *common ancestor* of both (Decoder-A's block is literally two of these stacked;
Decoder-B's block is two of these plus a residual connection). This neutral block is used to train the
shared front-end.

**Reasoning:** if the front-end were pretrained using, say, Decoder-A's own block, its features could
become subtly tuned to that block's inductive biases, biasing the later comparison in that decoder's
favor. Training with the neutral common-ancestor block avoids this without requiring a separate,
unrelated pretext task.

### 2.2 Per-fold pretraining (no label leakage)
Unlike the label-free SimCLR stage, this stage trains against real 3D targets, so it **must** respect
the cross-validation split — it is repeated once per fold, each time seeing only that fold's training
instances. The resulting front-end checkpoint and the fold's train/val/test instance list are both
written to disk, keyed by fold number, so every later stage (decoder training, comparison) can load an
**identical** split for that fold.

### 2.3 Diagnostics reused from step 1
The same axis-alignment and per-axis-variance checks are re-run here (they must pass again before any
training happens), plus a NaN diagnostic that walks the forward pass under both full precision and
mixed precision to localize exactly where non-finite values would first appear if training ever
destabilizes.

### 2.4 Local vs full-scale
This notebook is deliberately run at a small resolution, a handful of instances, and 2 epochs — its
only purpose is to prove the full pipeline executes end-to-end on CPU before committing to the
full-scale GPU run.

---

## Step 3 — Decoder pipeline (`03_decoder_pipeline.ipynb`)

**Goal:** attach one of two interchangeable decoder architectures to the front-end and train/evaluate
it, with the choice of decoder controlled by a single flag so the same notebook produces both runs.

### 3.1 Shared decoder skeleton, one differing block
Both decoders use identical wiring: they consume the encoder's four cube feature levels, upsample
step-by-step with the matching encoder feature concatenated as a skip connection (a standard
encoder-decoder shape), then a super-resolution head grows the result up to the full target
resolution. Channel counts taper as resolution increases, keeping memory affordable at the top
resolution.

The only thing that differs between the two candidates is the **convolutional block** used at every
stage:
- **Decoder-A block** — two conv+norm+activation layers, no shortcut.
- **Decoder-B block** — two conv+norm+activation layers *plus* a residual connection (helps gradient
  flow in deep volumetric networks).

**Reasoning for this framing (stated explicitly in the notebook):** because the encoder, skip
connections, resolution schedule, and loss function are all shared, this isolates the *decoder
conv-block* as the only comparison variable. The notebook is explicit that this is **not** a canonical
comparison of two full published architectures — Decoder-B's original design also includes its own
downsampling path and objective function, which live in the shared parts here. The honest framing is
"a controlled decoder-block ablation inside a shared encoder-decoder," not "architecture A vs
architecture B" — and the write-up is instructed to state it that way.

### 3.2 Two front-end "regimes"
Every decoder is trained under two conditions:
- **Frozen** — the entire pretrained front-end is frozen; only the decoder trains. This is the strict,
  apples-to-apples comparison (both decoders see byte-identical input features).
- **Finetuned** — the whole stack (encoder + fusion + lift + decoder) trains jointly. This is more
  realistic in terms of achievable capacity, but is explicitly *not* a pure decoder-only ablation,
  since the encoder can now also adapt differently per decoder.

Reporting both regimes lets the write-up distinguish "does the decoder block matter on its own" from
"does the decoder block matter once everything can adapt together."

### 3.3 Cross-validation, not a single split
A 5-fold, instance-stratified cross-validation scheme (mirroring the CV setup used in comparable
published reconstruction work) ensures every instance is used as a held-out test case exactly once.
This matters because the irregular cohort has a small number of instances — a single fixed train/test
split would leave only a couple of instances to evaluate that cohort on, which is not enough to draw a
conclusion from. Pooling every fold's held-out predictions gives a score for *every* instance in that
cohort.

### 3.4 Loss and metrics
- **Loss:** a 50/50 blend of binary cross-entropy and soft-overlap (Dice) loss, averaged across the
  four structural-component channels. The target structure occupies only a small fraction of the
  volume, so pure cross-entropy would be dominated by trivially-correct background voxels; adding the
  overlap term directly optimizes for the content that actually matters. (The blend was added on top
  of the reference implementation's overlap-only loss to stabilize early training.)
- **Overlap metrics:** Dice and IoU on the thresholded prediction, computed per component and
  averaged.
- **Surface (boundary) metrics:** 95th-percentile symmetric surface distance and average symmetric
  surface distance, both converted to physical distance units. **Reasoning:** overlap metrics measure
  regional similarity but can be insensitive to local boundary error, and the phenomena this project
  cares about most (fragmentation, displacement) are fundamentally boundary phenomena — so distance
  metrics are tracked specifically because they are the ones that actually move when boundary quality
  changes, matching how comparable published work reports results.
- All metrics are computed per instance and reported both **overall** and **split by cohort**
  (standard vs. irregular), since the split-by-cohort numbers are what the research question turns on.

### 3.5 Training loop and checkpointing
Standard train/validate loop with cosine learning-rate decay, gradient clipping, and non-finite-loss
guarding (a batch that produces NaN/Inf loss is skipped rather than corrupting the optimizer state).
Every epoch writes a "last" checkpoint (for resuming), a periodic numbered checkpoint (so any epoch can
be revisited later), and a "best" checkpoint whenever validation overlap score improves — each storing
the full run configuration alongside the weights so it can be reloaded unambiguously later (including
by the interactive viewer in step 5).

### 3.6 Post-training validation ("smoking-gun" checks)
Beyond the axis/variance checks inherited from steps 1–2, a third check inspects every prediction on
the held-out set for two failure modes that would otherwise pass silently: **collapse** (predicting
almost nothing, or almost everything) and **residual extrusion** (still constant along an axis despite
passing the earlier checks). A visual showcase then renders surface reconstructions (via marching
cubes) and orthogonal mid-slice overlays for the worst, median, and best-scoring instances plus a
representative irregular instance, annotated with their metric values, and saves them as images for
the write-up.

### 3.7 Optional capacity check
A small opt-in diagnostic (disabled by default) overfits just the decoder on two instances to confirm
the architecture has enough capacity to represent all four structural-component channels
independently — a sanity check that a channel isn't structurally unlearnable, separate from the main
cross-validated comparison.

---

## Step 4 — Decoder comparison (`04_decoder_comparison.ipynb`)

**Goal:** pool every fold/regime/decoder run from step 3 into the headline comparison table and test
whether any observed difference is statistically meaningful. This notebook does no training — it only
aggregates results already on disk.

### 4.1 Pooling
Every per-instance metric file across all folds is concatenated. Because cross-validation guarantees
each instance is held out exactly once, this pooling naturally produces exactly one score per instance
per decoder/regime — not five (one per fold) — which is what makes the small irregular cohort
statistically usable at all.

### 4.2 Headline table
Mean ± standard deviation of every metric, broken out by regime (frozen/finetuned), by group
(overall / standard cohort / irregular cohort), and by decoder.

### 4.3 Paired significance test
A two-sided **Wilcoxon signed-rank test** (the standard non-parametric test for comparing two models
on the *same* held-out instances without assuming a normal distribution of scores) pairs each
instance's two decoder scores together, per metric, per regime, per group. Because four metrics are
tested within each regime/group combination, **Holm-Bonferroni correction** is applied across that
family to control the false-positive rate from testing multiple metrics at once.

**Reasoning for this specific approach:** with a fairly small number of irregular instances, a single
train/test split would be statistically uninformative — cross-validation plus a paired non-parametric
test is treated as the minimum rigor needed to make any defensible claim about which decoder performs
better, following the same reporting pattern used in comparable published reconstruction-evaluation
work.

### 4.4 Per-component breakdown
The same mean±std and paired-test machinery is repeated per individual structural component (rather
than averaged across all four), since the two smaller components are structurally the hardest to
reconstruct and worth inspecting separately from the aggregate score.

### 4.5 How results feed the conditional stretch goal
The notebook states explicitly how to read the outcome: if scores on the irregular cohort remain low
under **both** regimes and the two decoders are statistically indistinguishable, the conclusion is
that the *decoder block* is not the bottleneck for that cohort — which is precisely the evidence
needed to justify (or not) moving on to a custom, irregularity-aware architecture, rather than
assuming it's needed up front.

---

## Step 5 — Interactive viewer (`05_decoder_ui.ipynb`)

A small interactive app for loading any saved checkpoint (any fold / regime / decoder / epoch) and
running inference on either a built-in held-out instance or an uploaded pair of projection images,
viewing the reconstructed 3D surface and mid-slice overlays. When a built-in instance has ground truth
available, it also reports overlap scores.

**Reasoning:** this exists for two purposes — (1) quickly revisiting any specific checkpoint without
re-running a notebook, and (2) supporting qualitative review of a separate real-world image cohort that
has no paired 3D ground truth, so its only possible evaluation is visual inspection rather than a
numeric score.

---

## Key cross-cutting engineering decisions

| Decision | Reasoning |
|---|---|
| Shared encoder, decoder-only difference | Isolates the variable actually under study; anything else shared would confound the comparison. |
| SimCLR pretrained once globally, front-end pretrained per fold | Self-supervised pretraining uses no labels (safe to share across folds); anything trained against real targets must respect fold boundaries to avoid label leakage. |
| Neutral common-ancestor block for front-end pretraining | Prevents the shared features from being implicitly biased toward one decoder's inductive bias. |
| Orthogonal back-projection lift (replacing extrusion) | The prior approach architecturally could not encode depth-varying structure, capping results regardless of decoder — this was the single highest-impact fix in the pipeline. |
| Two front-end regimes (frozen / finetuned) | Separates "does the decoder block matter in isolation" from "does it matter once the whole stack can adapt." |
| Per-component multi-channel target instead of single-channel occupancy | Removes an intensity-based confound and enables per-structure evaluation. |
| Structure-focused input mask with a safety gate | Improves signal-to-noise without risking silent loss of a real disconnected sub-region. |
| k-fold cross-validation + paired Wilcoxon + Holm-Bonferroni | The minimum statistical rigor achievable given a small irregular cohort; a single split would not support a defensible conclusion. |
| Overlap **and** surface-distance metrics | Overlap metrics alone can miss boundary-level differences, which is exactly what matters most for the hardest cohort. |
| Explicit "ablation, not canonical architecture comparison" framing | Scientific honesty: the shared parts of the pipeline mean this measures one design axis (the conv block), not two complete named architectures. |

---

## Status / what's still pending

- All five notebooks currently run as **local smoke tests** (small resolution, few instances, CPU,
  1–2 epochs) — their purpose is solely to prove the pipeline is correct end-to-end.
- The full-scale run (much higher resolution, full dataset, GPU) is mirrored notebook-for-notebook
  under `HPC/HPC_notebooks/modeling_HPC/` and is still pending execution.
- The grid of runs needed for the full comparison is: **5 folds × 2 regimes × 2 decoders** — each
  producing its own checkpoint directory and per-instance metric file for
  `04_decoder_comparison.ipynb` to pool.
- Whether the conditional custom-architecture stretch goal is pursued depends entirely on the outcome
  of that pooled comparison on the irregular cohort (see §4.5).
