# Stage 2 agent-run handoff records (Claude lane)

Every Stage 2 task — and in particular every HPC run — writes a record here under
`stage2/<task_id>/`. This directory is the audit trail the Codex-run Agent-N gate reviews.
Submission of a notebook or the existence of an output file is **never** evidence of success;
a dependent Stage 2 step resumes only after a recorded `PASS`.

## One folder per task: `stage2/<task_id>/`

Recommended `task_id` scheme: `s2<step>_<what>_fold<k>` (e.g. `s2_1_fcmae_fold0`,
`s2_3_decoder_overfit`).

Contents:

| File | Purpose |
|------|---------|
| `handoff.md` | The verdict record (template below) |
| `task_brief.md` | What the user was asked to run on HPC (path, cells/config, env, expected outputs, pass/fail indicators) |
| `config.json` | The exact run configuration used |
| `summary.csv` / `summary.json` | Metrics the notebook produced |
| `*.png` | QA figures / overlays / learning curves |
| `traceback.txt` | Exception log, if the run failed |
| `hashes.txt` | SHA-256 + path + size for large `.nii.gz` / `.pth` left on HPC |
| `task_brief_for_codex.md` | Only if Stage 2 needs a Stage 1/0 change — filed, not self-edited |

Large artifacts (`.nii.gz`, `.pth`) may remain on HPC; record their path/size/version/SHA-256
in `hashes.txt`. Small evidence is copied into `HPC/HPC_results/<task_id>/<run_id>/`.

## `handoff.md` template

```
# Stage 2 handoff — <task_id>

- Step: <S2.1 FCMAE | S2.2 front end | S2.3 decoder | S2.4 metrics>
- Fold: <k or n/a>
- Environment: <local CPU | HPC GPU>, working dir: <path>
- Notebook: <path>
- Expected outputs: <paths>

## Evidence returned
- Executed notebook / log: <path or attached>
- Config: config.json
- Summary: summary.csv|json
- QA figures: <list>
- Resource usage: GPU model / requested RAM / wall time / peak GPU+host memory
- Job ID + bulk-output path: <...>
- Hashes: hashes.txt

## Pass/fail indicators checked
- <indicator> -> <observed>

## Verdict: PASS | RETRY | BLOCKED
- Reviewer (Agent N): Codex
- Reason / next action: <...>
- Date: <YYYY-MM-DD>
```
