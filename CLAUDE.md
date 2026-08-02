# CLAUDE.md

## Project

3D Knee Reconstruction: Comparing **3D U-Net** vs **V-Net**.
Capstone 2 — Chan Zheng Shao, Sunway University. Supervised by Assoc. Prof. Ts. Dr Lee Yun Li.

### Behavioral Contract

1. Think Before Coding
   Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

State your assumptions explicitly. If uncertain, ask.
If multiple interpretations exist, present them - don't pick silently.
If a simpler approach exists, say so. Push back when warranted.
If something is unclear, stop. Name what's confusing. Ask. 2. Simplicity First
Minimum code that solves the problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code.
No "flexibility" or "configurability" that wasn't requested.
No error handling for impossible scenarios.
If you write 200 lines and it could be 50, rewrite it.
Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

3. Surgical Changes
   Touch only what you must. Clean up only your own mess.

When editing existing code:

Don't "improve" adjacent code, comments, or formatting.
Don't refactor things that aren't broken.
Match existing style, even if you'd do it differently.
If you notice unrelated dead code, mention it - don't delete it.
When your changes create orphans:

Remove imports/variables/functions that YOUR changes made unused.
Don't remove pre-existing dead code unless asked.
The test: Every changed line should trace directly to the user's request.

4. Goal-Driven Execution
   Define success criteria. Loop until verified.

Transform tasks into verifiable goals:

"Add validation" → "Write tests for invalid inputs, then make them pass"
"Fix the bug" → "Write a test that reproduces it, then make it pass"
"Refactor X" → "Ensure tests pass before and after"
For multi-step tasks, state a brief plan:

1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
   Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

These guidelines are working if: fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

### Goal

Conduct a comparative baseline between **3D U-Net** and **V-Net** on 3D reconstruction across all three datasets. Both models share the same encoder (ConvNeXtV2 + cross-attention fusion from Lai's pipeline); only the decoder differs.

1. **3D U-Net** — baseline from Lai's senior project (code in `references/01–10`)
2. **V-Net** — comparison model

**Conditional stretch goal**: If the 3D U-Net or V-Net baselines perform poorly at reconstructing fracture-sensitive cases, tailor a custom architecture (building on U-Net or V-Net) targeted at fracture-aware 3D reconstruction. This is only in scope IF the baselines demonstrate poor fracture reconstruction — do not treat it as committed work.

### Conditional Stretch Goal — Fracture-Aware Architecture

If both baseline models (3D U-Net and V-Net) perform poorly at reconstructing fractured knee
anatomy, the next step is to improve the architecture of whichever model handles fracture cases
better, making it fracture-aware. This is **conditional** — only pursued if baseline results
warrant it.

**Training data considerations**: When designing training/evaluation splits for the baseline
comparison, ensure balanced representation of healthy vs fractured cases to avoid underfitting
on fracture patterns or overfitting on healthy anatomy. This is especially important if the
conditional fracture-aware architecture is later needed.

### Evaluation Strategy

| Dataset            | Type        | Evaluation   | How                                                      |
| ------------------ | ----------- | ------------ | -------------------------------------------------------- |
| VSD (healthy)      | CT → DRR    | Quantitative | Reconstruct from DRRs, compare against ground-truth CT   |
| Fractured (Ruikar) | CT → DRR    | Quantitative | Same as VSD                                              |
| Regen (clinical)   | Real X-rays | Qualitative  | Visual assessment by actual doctors (no 3D ground truth) |

### Reference Code (`references/`)

Notebooks `01`–`10` are **Lai's original 3D U-Net implementation**. Do NOT modify them.
They serve as the baseline codebase to build upon and compare against.

Pipeline: SimCLR pre-training → ConvNeXtV2 encoder → cross-attention bi-planar fusion → 3D U-Net decoder.

## Workflow Rules

### Notebook-First Development

All pipeline code must be written in `.ipynb` notebooks with proper documentation.
The user will review notebook results before deciding whether to convert them to `.py` modules.
Do NOT create standalone `.py` scripts for pipeline steps — always use notebooks first.

## Environments

### Local (Windows)

- **OS**: Windows 11, PowerShell
- **Project root**: `C:\Users\Chan Zheng Shao\OneDrive\Desktop\Github Repo\TestProject\`
- **Notebook root**: `TestProject/notebooks/`
- **Target volume**: 256³
- **Device**: CPU

### HPC (Sunway University)

- **Platform**: AWS Linux, accessed via Open OnDemand (browser, code-server)
- **Python**: 3.12, venv at `.venv/`
- **Working directory**: `/home/project/xray2mesh/Marcus_Chan_Zheng_Shao_CP2_24020059/`
- **HPC structure**:
  ```
  Marcus_Chan_Zheng_Shao_CP2_24020059/
  ├── .venv/
  ├── data/
  │   ├── interim/merged_vsd/     # VSD_010/015/016/017/019_merged.nii.gz
  │   └── raw/
  │       ├── fractured/
  │       ├── Regen_XRays/
  │       └── VSD_Dataset/        # 001,002,005,006,010,014,015,016,017,019,023,z001,z004,...
  ├── notebooks/                  # HPC pipeline notebooks
  ├── requirements.txt            # DO NOT use — contains pywin32 (Windows-only)
  └── temp_reqs.txt
  ```
- **Requirements**: Use `requirements_hpc.txt` from the local digital twin (pywin32 removed, TotalSegmentator added)
- **Target volume**: 256³
- **Device**: `cuda` (GPU available)
- **TotalSegmentator**: `fast=False` (full model on GPU)

### Local HPC Digital Twin (`TestProject/HPC/`)

Mirrors the HPC structure for local development before uploading:

- `TestProject/HPC/HPC_notebooks/pre_processing_HPC/` → maps to HPC `notebooks/`
- `TestProject/HPC/requirements_hpc.txt` → the corrected requirements for HPC

**Naming convention**: local uses `HPC_notebooks/pre_processing_HPC/`; HPC uses `notebooks/`.

When writing HPC notebooks, always use Linux paths rooted at the HPC working directory.
Do NOT use PowerShell syntax in HPC notebooks — use POSIX shell.

## Confirmed Parameters (CT Preprocessing)

- **Spatial resampling**: 0.5mm isotropic
- **HU bone window**: [-450, 1050]
- **Canonical orientation**: LPS (Left-Posterior-Superior) end-to-end
- **Intermediate resampling**: 0.5mm isotropic; the stored 200mm/256 grid is 0.78125mm isotropic
- **Target volume**: 256³ for 3D volumes / 256×256 for 2D images — unified across local and HPC
- **DRR method**: DiffDRR (unified for healthy + fractured)
- **Project layout**: Cookiecutter Data Science (`testproject/` package)

Note: Regen X-rays are real clinical images — they do NOT go through the CT preprocessing
pipeline. They require their own image preprocessing (resize, normalize, etc.).

## Dataset Notes

### VSD (Healthy)

- 11 normal subjects: 8 single-folder DICOM + 3 multi-folder merged (010, 015, 017)
- 20 z-prefix full-body CT subjects (z001–z066) in the source study — processed via `vsd_z_crop.ipynb`; z057 Left/Right are excluded for metal artefacts
- Multi-folder subjects merged in `data/interim/merged_vsd/` before knee cropping
- Scans are bilateral — use connected-component analysis to separate left/right legs
- Knee crop margin: +-100mm around knee center
- Output: `data/raw/healthy/VSD.{id}/` per-case folders
- **Excluded (TKR)**: the user-approved surviving knees for z050 and z063 are anatomical Right;
  intrinsic LPS fibula/tibia evidence confirms this. Their affected anatomical Left knees are excluded
  for severe metal artefacts. Raw source files remain unchanged.

### Fractured (Ruikar)

- Case4 and Case10 excluded (scout images, 1 Z-slice each; MIN_Z_SLICES=10)
- Case8 excluded for fracture-fixation metal hardware
- Already knee-region scans — no bilateral separation needed
- Bimodal Z-spacing: 0.7mm (8 cases) and 3.0mm (6 cases)

### Regen (Clinical X-rays)

- 30 patient folders with real bi-planar X-rays (.dcm) and annotations (.ann)
- ~2 X-ray views per patient (AP + lateral)
- No paired CT volumes — qualitative evaluation only
- Contains patient names in folder/filenames — handle with care

## Foundation Experiment Contract

- Final quantitative cohort: 58 healthy knees + 13 fractured knees = 71.
- Reconstruction target: four binary channels in `[femur, tibia, patella, fibula]` order.
- Encoder policy: five fold-specific FCMAE P1 + cross-view P2 checkpoints, trained without test-fold subjects.
- Shared front end: train through a neutral head per fold, discard the head, then freeze encoder/fusion/lift for both decoders.
- Raw Regen PII is not modified. Generated manifests, figures, logs, and handoffs use study IDs only.
- Full U-Base/V-Base cross-validation is blocked until the Stage 2 independent QA gate passes.

## HPC Result Handoff

Heavy preprocessing and modelling notebooks may run through Open OnDemand. After every HPC run,
the responsible agent must ask the user for the executed notebook or log, configuration, summary
CSV/JSON, QA figures, resource usage, job/output paths, and hashes for large outputs. Submission or
file existence is not evidence of success. Dependent work resumes only after a recorded PASS verdict.
