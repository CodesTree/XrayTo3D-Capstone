# Decision record: per-bone retention denominator

- Decision: measure preprocessing retention from each STL rasterized on its version-matched, oriented 0.5 mm CT grid before the ROI crop and fixed-FOV operations.
- Reason: the gate is intended to detect anatomy removed by this preprocessing pipeline; voxels outside the CT acquisition grid cannot be retained by any downstream crop.
- Supporting evidence: the focused CT-STL overlay audit established that the intended STL anatomy is covered and correctly assigned before this task.
- Limitation: this metric does not independently certify mesh anatomy lying outside the acquired CT grid. A newly discovered CT-STL coverage problem invalidates the retention result and requires a new target-integrity review.
- Visual safeguard: the eight previously warned fractured cases require user review of complete-pre-FOV versus removed-voxel overlays even when the numeric gate passes.

