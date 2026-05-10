# CLAUDE.md

## Project

3D Knee Reconstruction + Fracture Segmentation MTL Framework.
Capstone 2 — Chan Zheng Shao, Sunway University.

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

## Confirmed Parameters

- **Spatial resampling**: 0.5mm isotropic
- **HU bone window**: [-450, 1050]
- **Orientation**: RAS (Right-Anterior-Superior)
- **Target volume**: Config-driven — 128³ (local) / 512³ (HPC)
- **DRR method**: DiffDRR (unified for healthy + fractured)
- **Project layout**: Cookiecutter Data Science (`testproject/` package)

## Dataset Notes

- VSD subject 010 excluded (scan too short, does not contain knee region)
- VSD scans are bilateral — use connected-component analysis to separate left/right legs
- Knee crop margin: +-100mm around knee center
