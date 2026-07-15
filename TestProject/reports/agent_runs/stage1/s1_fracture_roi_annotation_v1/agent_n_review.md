# Agent N independent fracture-target review

## Verdict

**PASS**

The Stage 1 fracture-target subgate is accepted. Certified AP/LAT DiffDRR generation is unblocked by this review. Full Stage 1 is still incomplete until the DRR evidence is generated, certified and included in the final independent Stage 1 gate review.

## Scope and evidence inspected

- Governing instructions: `AGENTS.md`, `.codex/agents/common_contract.md`, `.codex/agents/roles/agent_n_independent_qa.md` and the task brief in this directory.
- Executed producer notebook: `TestProject/notebooks/data_management/01_fracture_roi_annotation.ipynb`.
- Cohort/status record: `TestProject/reports/manifests/fracture_roi_status_v1.csv`.
- Source masks, aligned masks, QC CSV, summary JSON, run configuration, three-plane overlays and evidence hashes under `TestProject/data/interim/fracture_roi_lps_256_v1/`.
- Producer handoff and the annotation/data-change note in this directory.

## Independent checks

- Cohort reconciliation found 13 unique expected Ruikar rows: 12 `verified_fracture` and one `no_visible_fracture`, with no missing, duplicate, pending or unexpected status rows.
- All 12 verified rows have a nonempty reviewer, review date and `visual_approval=approved`. The Case2 no-visible-fracture decision is also explicitly approved.
- The executed notebook has no error output. Its final execution reports 12 verified replay completions, zero failure rows, 12 approved verified rows and readiness for independent review.
- The notebook's five synthetic replay assertions completed successfully.
- All 28 entries in `qa_v1/evidence_sha256.txt` exist and independently recompute to the recorded SHA-256 values.
- Every verified source mask is nonempty and strictly binary with values `[0, 1]`.
- Every verified source mask exactly matches its recorded source CT in size, spacing, origin and direction. Case3 and Case16 correctly use the cast-cleaned NIfTI sources.
- Every aligned mask is nonempty, strictly binary, 256 cubed at 0.78125 mm isotropic, and exactly matches its predrr image grid. All four bone targets also match the predrr grid.
- Bone-union intersections were independently recomputed for all 12 verified cases and exactly match the QC CSV; all are nonzero.
- Full transform replay was independently repeated for Case11, Case12, Case13, Case14, Case15, Case3 and Case9. Every replay retained exactly 100 percent of the 0.5 mm ROI through crop and fixed FOV, reproduced the QC counts, and produced an aligned array exactly equal to the saved mask. This sample deliberately includes all documented resampling/trimming exceptions.
- All 12 generated axial/coronal/sagittal centroid overlays were inspected. They show the ROI in the knee region with agreement to the surrounding four-bone target contours; the user's case-by-case anatomical approvals are recorded in the status manifest.

## Case-specific adjudication

- **Case2_PartLeft:** `no_visible_fracture` is internally consistent. No source or aligned fracture mask exists, which is the expected state for this status; it is not counted among verified replay rows.
- **Case9_PartRight:** the hand annotation originated on a 0.7 mm reconstruction and was resampled with nearest-neighbour interpolation in the shared physical frame onto the recorded 512 x 512 x 91, 3.0 mm source grid. The canonical source mask now exactly matches that CT, independently replays with 100 percent retention, exactly reproduces the saved aligned mask and intersects 113,129 target-bone voxels. The remaining limitation is unavoidable 3.0 mm superior-inferior quantisation unless Case9 preprocessing is redesigned end-to-end; no such redesign is authorised by this gate.
- **Case12_PartRight and Case3_PartLeft:** the documented out-of-FOV disconnected paint voxels were removed before the final run. Independent replay confirms that the current masks lose zero voxels through crop/FOV and exactly reproduce the saved aligned outputs. Small in-FOV disconnected components remain: 58 aligned voxels in each case, with each component at most 5 voxels for Case12 and at most 7 voxels for Case3. They are negligible relative to the main ROIs and were accepted through the user's full-volume visual review, so they do not overturn this gate. They must not be silently morphology-cleaned after this review.
- **Case15_PartRight:** the documented out-of-FOV paint cleanup also replays with 100 percent retention. Its aligned ROI has two substantial components (63,381 and 35,739 voxels), consistent with a multi-fragment/multi-bone fracture ROI rather than residual speck noise.

## Evidence identity

The producer hash list covers aligned masks, overlays, QC/config/summary files and the status manifest, but not the manual source masks. Agent N therefore records the accepted source-mask identities here. Any change to one of these files invalidates this review and requires regeneration and re-review.

```text
f406b56d60ee0090ed0354c623338aa7f15aa66c76295e947937908c164bb9a8  Case1_PartLeft_fracture_roi_source.nii.gz
3ea4c071ff0c6de08e80f43bfad5ed3e247d77a38adb1f4f323e18d36907891a  Case11_PartRight_fracture_roi_source.nii.gz
90940ffc4560df8ca312906f960af105fd92e478c077a93ab4f7eaa6e0003b2d  Case12_PartRight_fracture_roi_source.nii.gz
f6a182b5f45889b2893eb330029732b6952bc9a97dd4617e98f1850198bb9361  Case13_PartRight_fracture_roi_source.nii.gz
de1348f5a7e0562d8627dd119e2cb2a47c2a38cff2d6f3a23eec40f824bc16e4  Case14_PartRight_fracture_roi_source.nii.gz
742ea8381e5e86ba4809f3ab9e486bc6fdd4cc36a9eb33e039d3f187786a0938  Case15_PartRight_fracture_roi_source.nii.gz
2b041122a09f58d8a08f3839aad1c68c7d0d82d616b055724cb99c60ccc61b1a  Case16_PartRight_fracture_roi_source.nii.gz
f02a046af151b7b57294e3b07fd6b015682b9d1142ffe15ce413acda4c5c0c73  Case3_PartLeft_fracture_roi_source.nii.gz
b3dcc3b24e2dc06a6e5008b81a51023ac0c0f401e86b828db18f545580d7a9f9  Case5_PartRight_fracture_roi_source.nii.gz
1b69e25d101901ec4fdea8d9aab7f951387d1c5cf817e913e941f428bd278048  Case6_PartRight_fracture_roi_source.nii.gz
dab61aa8d334b6327b08b3698fb687ef7edcdc092c41d6bbe772935d45e69ba1  Case7_PartRight_fracture_roi_source.nii.gz
4068876694a06d262115be7691fe1576e3c842bb9265d93731304860b9d2a013  Case9_PartRight_fracture_roi_source.nii.gz
```

Key review inputs at decision time:

```text
1e9b115213e731d15eb88a026c9e7fbeb34faaf54baacc976df27fe0ec7539fb  01_fracture_roi_annotation.ipynb
8763b97eb83d6218ca5263a12701172ca2afd7380cd1fd46f585c2026484ec6c  fracture_roi_status_v1.csv
c3fc6ca9eecaf8bed7ce6c0d3b3214c306b5b0d27e33740dae5ef6e69a5a0828  fracture_roi_qc_v1.csv
2b5849b79ccb4003be7f4ccb3fc5024c57cfc32f5c2d766238542f59e76fe1ea  fracture_roi_summary_v1.json
aca8835a5dbc70368d6521c94f335ffaebb9e55074f2049269873cc45a37fb31  run_configuration.json
f6915bee6d36eb24062f77501e62f1ec53b010ae023827f051381a5f889c0745  evidence_sha256.txt
```

## Limitations and downstream controls

- The fracture ROIs are user-authored spatial evaluation regions, not a fifth reconstruction channel and not an independent radiologist diagnosis.
- Three-plane centroid overlays cannot expose every distant single-voxel component; the user's full-volume inspection is the anatomical approval record. Downstream fracture-local metric code must state whether it uses raw ROI voxels, connected components or bounding regions. Any filtering policy must be explicit, versioned and applied identically across models.
- The Case9 3.0 mm quantisation caveat must remain in downstream provenance.
- AP/LAT DRR generation may now start from the already certified predrr CTs. DRR generation must not modify these accepted source/aligned ROI masks, and its returned configuration, paired-view inventory, QA images and hashes require their own independent review.
