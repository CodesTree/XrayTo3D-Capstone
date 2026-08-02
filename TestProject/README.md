# Bi-planar Knee Reconstruction

Research pipeline for reconstructing four knee bones—femur, tibia, patella, and fibula—from paired AP and lateral digitally reconstructed radiographs (DRRs). The project covers CT preprocessing, geometry and laterality validation, per-bone target generation, fold-isolated representation learning, controlled decoder comparison, and out-of-fold evaluation.

> **Research use only.** This repository is an experimental study pipeline and does not make a clinical-utility claim.

## Study design

The canonical pipeline:

1. Normalizes CT data to LPS orientation and a fixed 200 mm field of view.
2. Resamples each volume to `256 × 256 × 256` at `0.78125 mm` isotropic spacing.
3. Generates paired AP/lateral DRRs and four-channel binary occupancy targets.
4. Trains a fold-isolated ConvNeXt V2 FCMAE encoder and a neutral shared front end.
5. Compares four decoder conditions in a controlled 2 × 2 design:
   - plain U-Net style
   - residual V-Net style
   - residual ReLU style
   - plain PReLU style
6. Evaluates out-of-fold Dice and average symmetric surface distance (ASSD), with subject-level grouping and pathology-stratified descriptive analysis.

The versioned study contract is defined in [`configs/data_contract_v1.json`](configs/data_contract_v1.json), while the fixed modeling protocol is defined in [`configs/baseline_protocol_v1.json`](configs/baseline_protocol_v1.json).

## Current status

- Canonical ready cohort: 71 knees (58 healthy and 13 fractured).
- Decoder evaluation: 71 knees from 43 subjects across five subject-grouped folds.
- The predeclared co-primary decision rule found the residual and activation main effects **inconclusive**; no decoder winner should be claimed from the current comparison.
- Fold 4 has predeclared front-end collapse evidence and is covered by a mandatory sensitivity analysis.
- The fractured subgroup is descriptive because of its limited sample size.

Generated evidence and machine-readable summaries are under [`reports/`](reports/), including the decoder comparison summary at [`reports/decoder_comparison/foundation_stage2_v1/comparison_summary.json`](reports/decoder_comparison/foundation_stage2_v1/comparison_summary.json).

## Requirements

- Python 3.12
- A CUDA-capable GPU is strongly recommended for training and 3D reconstruction
- JupyterLab or another Jupyter notebook environment
- Access to the source medical-imaging datasets; raw data is not included in this repository

The environment includes PyTorch, MONAI, ITK/SimpleITK, DiffDRR, timm, PyVista, scikit-image, and supporting scientific Python packages.

## Installation

From this directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the pinned environment and the local package:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the cluster environment, use [`HPC/requirements_hpc.txt`](HPC/requirements_hpc.txt) instead. Confirm that the installed PyTorch build matches the CUDA version available on the target system.

## Data layout

The package resolves all paths relative to this project directory:

```text
data/
├── raw/          # Original, immutable source data
├── external/     # Third-party data
├── interim/      # Resampled volumes, DRRs, targets, and ROIs
└── processed/    # Final modeling inputs
```

Do not commit identifiable source data, credentials, or local path configuration. Environment variables may be placed in a local `.env` file, which is ignored by Git.

The canonical generated artifacts are versioned as follows:

| Artifact | Version |
|---|---|
| Pre-DRR volumes | `predrr_lps_256_v1` |
| Per-bone targets | `gt_per_bone_lps_256_v1` |
| DRRs | `DRRs_diffdrr_lps_256_v1` |
| Fracture ROIs | `fracture_roi_lps_256_v1` |

Legacy binary-union and TSDF targets are explicitly excluded from the current protocol.

## Running the pipeline

The implemented research workflow is notebook-driven. Run notebooks in stage order and review each validation gate before proceeding.

### 1. Data governance and validation

Start with [`notebooks/data_management/`](notebooks/data_management/):

1. `00_manifest_and_reconciliation.ipynb`
2. `01_fracture_roi_annotation.ipynb`
3. `02_lps_geometry_validation.ipynb`
4. `03_notebook_environment_parity.ipynb`
5. `04_contract_validation.ipynb`
6. `05_vsd_merge_laterality_audit.ipynb`
7. `06_artifact_inventory.ipynb`
8. `07_vsd_cohort_laterality_audit.ipynb`
9. `08_vsd_focused_ct_stl_target_overlay_audit.ipynb`

### 2. Preprocessing

Use [`notebooks/pre-processing/`](notebooks/pre-processing/) for dataset-specific cleanup, knee cropping, canonical resampling, orientation checks, and paired DRR generation. The primary terminal stages are:

- `predrr_preprocessing.ipynb`
- `orientation_validation.ipynb`
- `drr_generation.ipynb`
- `data_augmentation.ipynb`

### 3. Modeling and evaluation

Run [`notebooks/modeling/`](notebooks/modeling/) in numbered order:

1. `00_gt_per_bone.ipynb` — build aligned four-channel targets
2. `01_encoder_pipeline.ipynb` — fold-isolated FCMAE and cross-view training
3. `01b_encoder_feature_audit.ipynb` — audit learned features
4. `02_frontend_pretrain.ipynb` — train the neutral shared front end
5. `03a_decoder_capacity_gate.ipynb` — verify decoder capacity and resume behavior
6. `03b_decoder_cross_validation.ipynb` — run five-fold decoder evaluation
7. `04_decoder_comparison.ipynb` — generate statistical comparisons and evidence
8. `05_decoder_ui.ipynb` — inspect reconstructions interactively
9. `06_metric_foundation.ipynb` — validate metric implementations

Mirrored cluster notebooks are available in [`HPC/HPC_notebooks/`](HPC/HPC_notebooks/). Keep local and HPC algorithms in parity; the data-management parity gate records the expected hashes and exceptions.

## Reproducibility safeguards

- Random seed: `42`
- Cross-validation: five-fold `StratifiedGroupKFold`, grouped by subject
- Validation fold: `(test_fold + 1) mod 5`
- Augmented samples inherit their parent subject's fold
- Test-subject access during encoder training is forbidden
- Decoder comparisons use a frozen, shared front end
- Model provenance and loaded checkpoint hashes are recorded per run
- Raw source files remain unchanged; generated Regen artifacts use study IDs only

The pretrained encoder configuration and license information are recorded in [`configs/foundation_stage2_pretrained_v1.json`](configs/foundation_stage2_pretrained_v1.json).

## Repository structure

```text
├── configs/       # Versioned data, training, pretrained-model, and laterality contracts
├── data/          # Local raw, interim, processed, and external data
├── HPC/           # Cluster environment and mirrored notebooks
├── models/        # Checkpoints, histories, and model provenance
├── notebooks/     # Exploration, data management, preprocessing, and modeling workflows
├── reports/       # Manifests, audits, metrics, figures, and statistical evidence
├── testproject/   # Shared Python paths and scaffolded command-line modules
├── Makefile       # Environment, linting, formatting, and cleanup helpers
├── pyproject.toml # Package metadata and Ruff configuration
└── requirements.txt
```

## Development commands

On systems with GNU Make:

```bash
make requirements  # install dependencies
make lint          # check formatting and lint rules
make format        # apply Ruff fixes and formatting
make clean         # remove Python cache files
```

The modules in `testproject/dataset.py`, `features.py`, and `modeling/` are currently Cookiecutter scaffolds; they are not replacements for the notebook research pipeline.

## License and data access

No project license is currently included. Unless a license is added, reuse and redistribution rights are not granted by default. Individual datasets and pretrained weights remain subject to their original terms; in particular, the configured ConvNeXt V2 FCMAE weights are recorded as CC BY-NC 4.0.