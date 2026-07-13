# Agent N Independent Review — Stage 0 Foundation Contract

- Review role: Agent N (independent QA)
- Scope: Stage 0 static contract, agent registry, artifact classification, and preprocessing local/HPC parity
- Verdict: **RETRY**
- Stage 1 HPC/data certification: explicitly not evaluated and not required for this Stage 0 verdict
- Stage 2 modeling files: not inspected

## Evidence that passed

- Registry contains 14 unique roles: `orchestrator`, A–L, and N; M is absent; G, I, J, and K are inactive.
- Every registry dependency resolves to a known role, every referenced role file exists, and no dependency cycle was found.
- All 14 role files contain the required mission, activation, inputs/dependencies, permitted outputs, local/HPC routing, tests/success, forbidden actions, HPC handoff, and downstream handoff sections.
- `common_contract.md`, `AGENTS.md`, `CLAUDE.md`, `data_contract_v1.json`, and `baseline_protocol_v1.json` consistently lock LPS, 58 healthy + 13 fractured = 71, five subject-grouped folds, per-fold FCMAE/cross-view pretraining, a frozen neutral-trained front end, and four channels ordered femur/tibia/patella/fibula.
- No stale authoritative `RAS` or `73` token was found in those contract files.
- Final-path prohibitions include fold 5, binary-union targets, TSDF targets, and neutral-smoke artifacts.
- Independent cell comparison passed for both preprocessing pairs: `predrr_preprocessing.ipynb` (18 cells; only cell 1 environment-tagged) and `drr_generation.ipynb` (13 cells; only cell 2 environment-tagged). All non-environment cells are identical between local and HPC twins.
- The artifact inventory has 382 unique rows, no missing classifications, and only the six allowed classes. The quarantine table marks 34 legacy/smoke artifacts non-selectable without deleting them.

## Retry findings

1. **Preprocessing documentation contradicts the locked implementation contract.** In both local and HPC `predrr_preprocessing.ipynb`, the configuration comment expands `LPS` as “Right-Anterior-Superior,” and the notebook header says the surviving z050/z063 contralateral knees are Left. Current LPS-centroid evidence instead infers the surviving files as Right and the excluded TKR sides as Left, with user approval still pending. Remove the incorrect expansion and keep the side statement explicitly pending until approval.
2. **The DiffDRR notebook header names a quarantined legacy output.** Both local and HPC `drr_generation.ipynb` document `data/interim/DRRs/`, while the active configuration writes `data/interim/DRRs_diffdrr_lps_256_v1/`. Update both notebook headers together so parity remains intact and HPC handoffs point to the versioned output.
3. **The artifact inventory is not a complete current snapshot.** Independent comparison found eight live files in its declared scope that are absent from `artifact_inventory_v1.csv`: the four repository-audit outputs themselves and four Stage 0/1 task-brief/handoff reports. Regenerate the inventory after current reports exist, or define and record an explicit snapshot timestamp/exclusion rule so self-generated and later agent-run evidence is deterministically classified without making the completeness claim stale.

## Required re-review evidence

- Updated local/HPC preprocessing notebook pairs with the misleading text corrected and parity still passing.
- Refreshed artifact inventory/summary/quarantine metadata, or an explicit auditable snapshot policy plus a zero-unexplained-missing comparison.
- Re-run static contract validation and provide its result.

These are Stage 0 documentation/audit-integrity corrections, not Stage 1 scientific or HPC blockers. Agent N must re-review the corrected evidence before Gate 0 is recorded as PASS.
---

## Agent N Re-review - 2026-07-13

- Superseding verdict: **RETRY**
- Scope remained Stage 0 only; Stage 2 modeling files were not inspected.

### Corrected evidence that now passes

- Both predrr twins now correctly expand LPS as Left-Posterior-Superior and describe the current z050/z063 centroid inference as excluded Left/surviving Right with user approval pending.
- Both DiffDRR twins now document `data/interim/DRRs_diffdrr_lps_256_v1/` and their active configuration uses the same path.
- Independent non-environment-cell comparison passes for all three registered pairs: predrr preprocessing, DiffDRR generation, and VSD merge/laterality audit.
- Core static contract checks still pass: LPS, 71 knees, five grouped folds, four ordered bone channels, per-fold FCMAE/cross-view, neutral frozen front end, and legacy/fold-5 prohibitions.

### Remaining retry findings

1. **The predrr summary is still stale in both twins.** The final markdown cell still documents `data/interim/predrr/healthy/` and `data/interim/predrr/fractured/` instead of `data/interim/predrr_lps_256_v1/{healthy,fractured}/`. It also says HPC uses 512 cubed and about 0.39 mm spacing, contradicting the locked unified 256 cubed, 0.78125 mm local/HPC contract. Correct that cell in both twins and preserve parity.
2. **The regenerated inventory omits two reusable reference checkpoints.** `artifact_inventory_v1.csv` has 391 unique classified rows, but independent comparison found these in-scope `.pth` files absent:
   - `TestProject/references/Lai's Capstone Project/convnextv2_finetuned_simclr.pth`
   - `TestProject/references/Lai's Capstone Project/src/utils/convnextv2_finetuned_xray.pth`

   The current inventory notebook scans checkpoints only under `artifacts/` and `models/`, even though the Stage 0 requirement covers all checkpoints and the earlier inventory classified these two reference weights as reusable. Add a targeted checkpoint scan for `references/` (without modifying the immutable reference files), then regenerate the inventory/summary/quarantine outputs.

### Evidence required for final Gate 0 review

- Corrected predrr summary cell in both twins and a passing parity report.
- Inventory containing the two reference checkpoint paths as reusable, with zero unexplained missing files in the declared scope.
- Passing static contract validation after regeneration.

Stage 1 pending user/HPC certification remains outside this verdict and is not a reason for RETRY.
---

## Agent N Final Re-review - 2026-07-13

- Final superseding verdict: **PASS**
- Scope: Stage 0 static foundation only; Stage 2 modeling files were not inspected.

### Independent verification

- Both predrr twins now document `predrr_lps_256_v1/{healthy,fractured}`, `TARGET_SIZE = 256` for local and HPC, and 200/256 = 0.78125 mm. No legacy predrr path, 512-HPC, or 0.39 mm claim remains in the reviewed twins.
- Both DiffDRR twins document and configure `DRRs_diffdrr_lps_256_v1`.
- Independent cell comparison passed for all three registered local/HPC pairs: predrr preprocessing, DiffDRR generation, and VSD merge/laterality audit. All non-environment cells match; environment-cell tags align.
- The artifact inventory contains 394 unique classified rows. Both reference FCMAE checkpoint files are present and classified `reusable`.
- Inventory metadata records UTC snapshot start, an explicit completeness rule, and zero unexplained missing paths. Independent reconstruction of the declared file scope found zero currently missing paths and zero pre-snapshot omissions.
- Independent static assertions passed for LPS, 58 healthy + 13 fractured = 71, 256 cubed at 0.78125 mm, ordered femur/tibia/patella/fibula channels, five subject-grouped folds, per-fold FCMAE/cross-view policy, frozen neutral-trained front end, and forbidden fold-5/binary/TSDF artifacts.
- Registry assertions passed: 14 unique IDs, all dependencies and role files resolve, G/I/J/K are inactive, and M is absent.

### Gate decision

Gate 0 is approved. The Stage 0 contracts, audit classification, quarantine policy, and preprocessing parity evidence satisfy the static foundation acceptance criteria.

This PASS does not certify Stage 1 data, laterality, manual ROI, versioned preprocessing, or HPC outputs. Those remain subject to their user-mediated result bundles and independent Stage 1 review. It also does not authorize Stage 3 baseline training; Gate 2 remains required.