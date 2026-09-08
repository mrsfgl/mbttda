# Validation report

Local delivery, 2026-09-08. Base revision: `189e39fa5e2d958616b389905f92b69f30f85ce4`.

## Completed checks

- Cloned the complete repository and created local branch `codex/modernize-mbttda`.
- Ran MISS_HIT 0.9.44 syntax/lint checks on all 87 first-party MATLAB source and test files; both checks passed.
- Applied consistent formatting to first-party MATLAB files.
- Inspected MAT-file headers to document the actual bundled array dimensions.
- Checked the upstream TTeMPS multiple-eigenvector return interface and current MathWorks Actions versions.
- Checked Git whitespace errors and reviewed source organization and path handling.
- Checked function/file name casing, duplicate first-party entry points, and all local documentation links.
- Compared Git content hashes for all 327 retained data/third-party assets against the baseline; all matched, including the five bundled MAT files.

Static analysis checks syntax and some local issues. It does **not** establish MATLAB runtime correctness, numerical equivalence, performance, or external dependency compatibility.

## Prepared tests

The MATLAB suite includes independent dense contraction references, hand-calculated scatter, TT-SVD reconstruction, orthogonality, threshold boundaries, storage accounting, validation isolation, compatibility signatures, method smoke tests, and result/plot integration.

Regression cases cover unequal test-set sizes, LDA class means, solver output ordering, independent parameter grids, missing presets, legacy storage units, metadata round trips, and legacy holdout/shape grouping. These are prepared tests, not completed MATLAB runs.

Run from the repository directory:

```matlab
setup_mbttda;
results = runtests('tests', 'IncludeSubfolders', true);
assertSuccess(results);
disp(table(results));
```

Optional TTNPE/BTT tests report skips when their external dependencies are absent. Required dependency failures are not skipped. Review both failed and incomplete test counts before describing a method as validated.

## Pending

| Check | Status |
| --- | --- |
| MATLAB R2024b test suite | Not run |
| Synthetic demo and generated figures | Not run |
| Linux and Windows GitHub Actions jobs | Configuration prepared; not published or run |
| External TTNPE / TT-Toolbox / TTeMPS integration | Not run |
| Full COIL, Weizmann, and other dataset sweeps | Not run |
| Cambridge / UCF-101 experiments | Original preprocessed data still needed |
| Numerical comparison with the baseline commit | Not run |
| Published accuracy, storage, and timing reproduction | Not claimed |

MATLAB execution and GitHub publication were explicitly deferred for this delivery. No MATLAB or Octave runtime was installed or launched. The optional static checker is in the ignored local `.tools` folder and is not a MATLAB project dependency.
