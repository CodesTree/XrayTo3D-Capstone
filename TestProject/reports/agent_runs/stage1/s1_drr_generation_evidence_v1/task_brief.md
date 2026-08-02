# Task brief: s1_drr_generation_evidence_v1

- Stage: 1
- Assigned role: Agent D - Orientation, Projection and Geometric Lifting
- Change class: methodological correction
- Objective: Produce a complete, independently reviewable numerical-QA and SHA-256 evidence bundle for the existing 71-case / 142-view DiffDRR cohort without rerendering DRRs.
- In scope: manifest-aligned AP/LAT inventory, metadata and geometry checks, NPY/PNG numerical integrity, paired-view distinction, NPY SHA-256 identities, configuration provenance, best-effort resource evidence, QA-figure presence, and local/HPC notebook parity.
- Out of scope: DRR rerendering, projection-parameter changes, CT/target/ROI modification, quantitative-manifest certification, and Stage 2 modelling.
- Predecessor gates and evidence: laterality PASS; predrr preprocessing PASS; four-bone target integrity PASS; fracture ROI PASS; executed local CPU and HPC CUDA DRR notebooks and their 142-row metadata tables are available.
- Inputs and versions: `quantitative_manifest_v1.csv`; `predrr_lps_256_v1`; `DRRs_diffdrr_lps_256_v1`; `diffdrr_siddon_lps_256_v1`; AP/LAT; 256 by 256 float32 NPY model inputs and uint16 PNG QA images.
- Permitted notebooks/outputs: local and HPC `drr_generation.ipynb` twins; `DRR_OUT_DIR/qa_v1/`; compact returned evidence under `HPC/HPC_results/s1_drr_generation_evidence_v1/<run_id>/`.
- Local or HPC: audit locally and on HPC; HPC generation outputs are canonical for the returned HPC evidence bundle.
- Assumptions requiring confirmation: none. Job ID, GPU model, requested RAM and wall time are best-effort informational fields and do not block the Stage 1 DRR gate.
- Success criteria: exactly 71 manifest samples and 142 unique views; 116 healthy and 26 fractured views; 71 AP and 71 LAT; zero missing, unexpected or duplicate keys; exact geometry metadata; all NPY/PNG and AP/LAT integrity checks pass; 142 NPY hashes are recorded; compact evidence is complete; independent Agent N review follows.
- Verification: compile all code cells; confirm evidence-cell source identity between twins; run synthetic negative cases; run the audit against existing local and HPC outputs; independently recompute sample hashes and metrics.
- Downstream role: Agent N - Independent Quality Assurance and Integration
- Overlap check: no concurrent worker may edit either DRR notebook while this task is implemented.
