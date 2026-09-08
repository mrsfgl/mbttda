# Modernization notes

Baseline: `189e39fa5e2d958616b389905f92b69f30f85ce4`.

## Structure and compatibility

- Moved first-party functions into responsibility-based source folders and added session-local setup.
- Consolidated experiment configuration, independent parameter sweeps, result loading, and plotting.
- Retained documented algorithm names, result fields, MDA signatures, and both `get_data` calling conventions.
- Corrected the three-branch filename casing.
- Replaced the incomplete plotting script and implicit workspace loads with explicit result loading.
- Original datasets and third-party files are unchanged; a tracked Finder metadata file was removed.

## Changes that can affect results

These are deliberate implementation repairs, not verified benchmark-equivalence claims.

| Change | Reason / regression coverage |
| --- | --- |
| Correct `lrnU` output unpacking | Objective histories were being used as timings; solver output and objective tests |
| Explicit local-core contractions | Replaces opaque contraction-order formulas with left/right environments; dense-map reference tests include singleton ranks |
| Finite convergence denominator and true previous iterate | Avoids division by zero in the initial stopping calculation; solver tests |
| Actual objective history when display is off | Quiet execution now records sweep objectives rather than mislabeling time values |
| Forward TT-SVD rewritten as a short loop | Preserves relative thresholding and supports two-dimensional inputs; reconstruction/threshold/orthogonality tests |
| Scatter normalization made explicit | Preserves the original covariance denominators while handling singleton dimensions; hand-computed tests |
| BTT return adapter and single-mode eigensolve | Retains all eigenvectors from TTeMPS; optional BTT tests and a dependency-free one-mode test |
| Dense LDA class dimensions | The old matrix input was treated as one class, eliminating between-class information |
| Dense LDA eigenvector ordering | Selects the smallest objective directions explicitly; retains the original reciprocal-eigenvalue rank heuristic with nonempty/full-rank guards |
| TTNPE eigenvector selection | Selects the smallest signed eigenvalues and handles full-rank requests; eigenvector tests |
| TTNPE stopping and one-mode inputs | Uses the real previous iterate, avoids zero division, and obeys the configured iteration limit; optional algorithm and one-mode solver tests |
| Training-only automatic MDA rank selection | The old loader estimated ranks from all samples; legacy loader behavior remains available |
| Explicit validation allocation | Records the final holdout and fold indices; isolation tests |
| Independent tau grids | Different method-grid lengths no longer produce an indexing error |
| Zero-noise random draws removed | Reproducible new runs; matching an old seed does not recover old splits |
| TTDA ablation axis mapping | Zeroing counts follow the sorted physical axes |
| TTDA test tensor layout | Test-set size is independent of training size; unequal-size and axis-order regression |
| TTNPE self-neighbor handling | Removes the actual sample index when identical observations create distance ties |
| Consistent elapsed-time stages | Timings include factor initialization and distinguish learning from classification |
| Element-based storage metric | Excludes MATLAB cell/container overhead, while preserving legacy fields |

The original scatter convention is a sum of per-class covariances for within scatter and covariance of class means for between scatter. Changing to unnormalized sums would change lambda's meaning; that change has not been made.

The contraction refactor keeps optimization within each branch, but its scheduling and stopping repair can change runtime and floating-point paths. Published speedups and exact output equivalence require fresh MATLAB measurements.

## Explicit boundaries

- Existing TT and MDA numeric presets remain, with documented missing values. No arbitrary three-branch grid has been invented.
- `main_App` now consistently performs classification for both argument layouts. Its old undocumented clustering-only `nmi` output is not retained.
- Historical TT and TTNPE adapters use 1-NN and report unsupported neighbor settings.
- The undocumented, unused `U2Ui_tau(...,f=0)` reverse-layout path now gives a clear error; no experiment in the original repository called it.
- Invalid old fixed-rank MDA tables fail explicitly rather than being silently rounded or resized.
- BTT retains its original algorithm-specific rank/sweep settings.
- Nonzero ablations retain each method's original placement in the computation; they are experimental settings, not a unified corruption model.
- STTM and raw-video preprocessing have not been reconstructed.
- Third-party code has not been refactored or relicensed.
