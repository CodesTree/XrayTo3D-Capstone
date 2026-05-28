# CLAUDE.md

## Project

3D Knee Reconstruction: Comparing **3D U-Net** vs **Implicit Neural Representation (INR)**.
Capstone 2 — Chan Zheng Shao, Sunway University. Supervised by Assoc. Prof. Ts. Dr Lee Yun Li.

### Goal

Reconstruct 3D knee anatomy from bi-planar X-ray images. Compare two decoder approaches:
1. **3D U-Net** — baseline from Lai's senior project (code in `references/01–10`)
2. **INR** — novel contribution (approach TBD: NeRF-based, occupancy networks, SDF, etc.)

Both share the same encoder (ConvNeXtV2 + cross-attention fusion from Lai's pipeline).

### Evaluation Strategy

| Dataset | Type | Evaluation | How |
|---------|------|------------|-----|
| VSD (healthy) | CT → DRR | Quantitative | Reconstruct from DRRs, compare against ground-truth CT |
| Fractured (Ruikar) | CT → DRR | Quantitative | Same as VSD |
| Regen (clinical) | Real X-rays | Qualitative | Visual assessment only (no 3D ground truth) |

### Reference Code (`references/`)

Notebooks `01`–`10` are **Lai's original 3D U-Net implementation**. Do NOT modify them.
They serve as the baseline codebase to build upon and compare against.

Pipeline: SimCLR pre-training → ConvNeXtV2 encoder → cross-attention bi-planar fusion → 3D U-Net decoder.

## Workflow Rules

### Notebook-First Development

All pipeline code must be written in `.ipynb` notebooks with proper documentation.
The user will review notebook results before deciding whether to convert them to `.py` modules.
Do NOT create standalone `.py` scripts for pipeline steps — always use notebooks first.

### CLAUDE.md Behavioral Contract

1. **State assumptions explicitly.** If a parameter, resolution, or design choice is ambiguous,
   name it and ask the user — do not pick silently.
2. **Simplicity first.** Write the minimum code that solves the immediate step. No speculative
   abstractions, no "future-proof" wrappers the user didn't ask for.
3. **Surgical changes.** When editing existing code, touch only what the current task requires.
   Match the user's existing style.
4. **Goal-driven execution.** Every task should have a stated success criterion and a verification
   step before moving on.

## Confirmed Parameters (CT Preprocessing)

- **Spatial resampling**: 0.5mm isotropic
- **HU bone window**: [-450, 1050]
- **Orientation**: RAS (Right-Anterior-Superior)
- **Target volume**: Config-driven — 128³ (local) / 512³ (HPC)
- **DRR method**: DiffDRR (unified for healthy + fractured)
- **Project layout**: Cookiecutter Data Science (`testproject/` package)

Note: Regen X-rays are real clinical images — they do NOT go through the CT preprocessing
pipeline. They require their own image preprocessing (resize, normalize, etc.).

## Dataset Notes

### VSD (Healthy)
- 11 normal subjects: 8 single-folder DICOM + 3 multi-folder merged (010, 015, 017)
- 20 z-prefix full-body CT subjects (z001–z066) — processed via `vsd_z_crop.ipynb`
- Multi-folder subjects merged in `data/interim/merged_vsd/` before knee cropping
- Scans are bilateral — use connected-component analysis to separate left/right legs
- Knee crop margin: +-100mm around knee center
- Output: `data/raw/healthy/VSD.{id}/` per-case folders
- **Excluded (TKR)**: z050 Right and z063 Right have total knee replacements — severe metal
  artifacts, unsuitable for training or testing. Their contralateral sides (Left) may still be usable.

### Fractured (Ruikar)
- Case4 and Case10 excluded (scout images, 1 Z-slice each; MIN_Z_SLICES=10)
- Already knee-region scans — no bilateral separation needed
- Bimodal Z-spacing: 0.7mm (8 cases) and 3.0mm (6 cases)

### Regen (Clinical X-rays)
- 30 patient folders with real bi-planar X-rays (.dcm) and annotations (.ann)
- ~2 X-ray views per patient (AP + lateral)
- No paired CT volumes — qualitative evaluation only
- Contains patient names in folder/filenames — handle with care
