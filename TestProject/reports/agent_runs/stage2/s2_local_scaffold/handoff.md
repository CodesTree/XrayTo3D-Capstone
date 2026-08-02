# Stage 2 handoff — s2_local_scaffold

- Step: Stage 2 bootstrap (S2.4 metric foundation + S2.3 GroupNorm swap + Claude-lane infra)
- Fold: n/a
- Environment: local CPU (Windows), venv `.venv/Scripts/python.exe`
- Owner (lane): Claude

## What was done (Claude-owned paths only)

1. `.claude/skills/SKILL.md` — refreshed from stale RAS/128³/INR framing to LPS / 256³ /
   U-Net-vs-V-Net / four-channel / notebook-first.
2. `.claude/agents/stage2-executor.md` — native launchable Stage 2 subagent (lane rules + HPC discipline).
3. `reports/agent_runs/stage2/README.md` — handoff record format.
4. `notebooks/modeling/tests/metric_foundation.ipynb` — **new**, S2.4 metric foundation.
5. `notebooks/modeling/03_decoder_pipeline.ipynb` + HPC twin — symmetric
   `BatchNorm3d → GroupNorm(8, C)` in DoubleConv + VNetResBlock; docstrings/markdown BN→GN.

## Evidence returned

- Metric notebook executed via nbconvert: exit 0, embedded output `=== 23/23 checks passed ===`.
  Covers empty-prediction/NaN handling, exact known-offset + mm-scaling, anisotropic-spacing
  guard (scalar 2.97 vs sampling-aware 9.24), CC agreement / min-gap / false-bridging, and
  macro/subject aggregation.
- Decoder smoke (model cell exec'd from the .ipynb, forward pass): both arms build and run —
  `plain_unet_style` (block=unet) and `residual_vnet_style` (block=vnet), 12 GroupNorm / 0
  BatchNorm each, output (1,4,16,16,16), ~8.5M params. JSON of both notebooks re-validated.

## Pass/fail indicators checked

- metric_foundation.ipynb -> 23/23 checks, no AssertionError -> PASS
- decoder builds+forwards with GroupNorm, zero BatchNorm remaining -> PASS
- lane isolation: only Claude-owned files changed -> PASS

## Verdict: PASS (local scaffolding)
- Reviewer (Agent N): Codex — pending (independent review of the Stage 2 gate).
- Remaining Stage 2 (this lane): arm-name labels + frozen-feature hash (sequence with S2.2
  frozen front-end run), then 256³ overfit/preflight on HPC. FCMAE (S2.1) and neutral front
  end (S2.2) stay gated on Codex's Stage 1 PASS.
- Date: 2026-07-13
