# Recovery notes — 2026-10-23

Retained the archived packed-sequence LSTM model. Removed top-level training side effects and corrected the derivative target for the training-only normalization scales.

## Shared changes

Added an explicit data root, deterministic CPU seeds, dependency declarations, synthetic demonstration, finite loss/gradient/prediction checks, held-out evaluation, checkpoint replay with protocol checks, and source provenance. Train/test preprocessing is separated. Archival code is not executed at import.

## Deviations and unverified parts

This runner selects one battery-data path, uses a compact model and capped windows, and has no ensemble, hyperparameter search or complete cross-validation. It does not reproduce all electronic-component case studies or published performance.

## 2026-10-24 release-readiness audit

- Specify Python 3.12, update the pip bootstrap, align default dependency installation with tested pins, and record a macOS transitive snapshot.
- Validate CLI values, reject empty/nonfinite normalization inputs, protect existing outputs, and record environment metadata.
- Add version-2 checkpoint protocols with input-file fingerprints. Older checkpoint protocols require retraining.
- Add regression tests and a CPU CI matrix; CI execution status is reported separately from local tests.
- Clarify runnable versus archival code, data confidentiality, reproducibility limits and release rights in the documentation.
- Remove constructor-level seed resets so the requested seed controls model initialization. This changes runs that previously ignored the seed.

## Follow-up release hardening — 2026-10-24

- Protocol v3 removes absolute machine paths, supports data-root relocation and binds replay to the repository and maintained source hashes. Recreate v1/v2 checkpoints.
- Add replay privacy/provenance regressions and dependency advisory checks on all three hosted operating systems.
- Avoid duplicate push/PR checks and cancel superseded workflow runs.
- Record explicit release decisions; no rights, dataset license or scientific approval is assumed.
