# Recovery notes — 2026-10-23

Retained the archived packed-sequence LSTM model. Removed top-level training side effects and corrected the derivative target for the training-only normalization scales.

## Shared changes

Added an explicit data root, deterministic CPU seeds, dependency declarations, synthetic demonstration, finite loss/gradient/prediction checks, held-out evaluation, checkpoint replay with protocol checks, and source provenance. Train/test preprocessing is separated. Archival code is not executed at import.

## Deviations and unverified parts

This runner selects one battery-data path, uses a compact model and capped windows, and has no ensemble, hyperparameter search or complete cross-validation. It does not reproduce all electronic-component case studies or published performance.
