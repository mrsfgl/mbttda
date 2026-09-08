# Experiment configuration and data

## Configuration

Create a complete, editable struct with `experiment_config(dataset)`, then pass it to `run_experiment`. Unknown fields fail rather than being ignored.

| Field | Default / meaning |
| --- | --- |
| `dataset` | `COIL`; also accepts `synthetic` and `custom` |
| `preset_family` | `tt`; `mda` selects the original Tucker layouts/grids |
| `methods` | `TTDA`, `TWTTDA`, `ThW` |
| `seed`, `repeats` | 0 and 10 |
| `data_dir` | Bundled `data` directory, or an explicit external directory |
| `tensor_shape`, `train_per_class` | Empty retains preset; training counts can be a row vector |
| `tau` | Empty retains independent method grids; a row vector overrides them |
| `tau_by_method` | Struct with uppercase method names, e.g. `TWTTDA = [0.2,0.1]`; overrides `tau` |
| `lambda` | Empty runs validation; scalar fixes all branches; three entries select branch-specific values |
| `validation_per_class` | 5; explicitly adjust for smaller training/test pools |
| `validation` | Search overrides: `candidates`, `repeats`, `maxiter`, `tau` |
| `ablations` | Empty retains preset; one count per physical mode in each row |
| `parameters` | Solver overrides such as `maxiter` and `error_tot` |
| `ranks` | Explicit MDA ranks; a vector is one setting, matrix columns are a sweep |
| `noise`, `scale` | 0 and 255; splitting divides loader output by `scale` |
| `save_results`, `output_dir` | true; an empty directory name generates a timestamped folder under `results` |
| `plot`, `figure_visible` | false and `on`; use `off` for noninteractive figures |
| `images`, `labels` | Only for `custom` data; not duplicated into result metadata |

Explicit MDA ranks replace threshold-based rank selection. Solver fields `I`, `tau`, `lambda`, and `nA` are set by the corresponding experiment settings. The original block solver retains its separate three-sweep/eight-rank limits.

A complete method collection is:

```matlab
config.methods = {'TTDA','TWTTDA','ThW','LDA','MPS', ...
                  'TTNPE','BTT','TWBTT','ThWBTT','CMDA','DGTDA'};
```

Check dependencies before choosing the external comparisons. TTDA, MPS, and BTT variants with multiple branches require enough physical modes.

## Original presets and paper differences

The TT preset values were extracted from revision `189e39fa5e2d958616b389905f92b69f30f85ce4`. MDA has its own historical layouts. The refactor does not silently change tensorizations to match the paper.

| Dataset | Repository TT shape | Repository MDA shape | Training samples/class |
| --- | --- | --- | --- |
| COIL | 4 x 4 x 4 x 4 x 4 x 4 | 8 x 8 x 8 x 8 | 20 |
| COIL3D | 64 x 64 x 8 | 64 x 64 x 8 | 3 |
| MNIST | 4 x 7 x 4 x 7 | 4 x 7 x 4 x 7 | 100 |
| Weizmann | 8 x 8 x 44 | 4 x 4 x 4 x 4 x 11 | 20 |
| Cambridge | 30 x 40 x 30 x 10 | Same | 4 |
| UCF101 | 5 x 6 x 5 x 8 x 5 x 10 | No historical MDA preset | 10:10:90 |
| YaleB | 30 x 40 x 20 | Same | 32 |
| GAIT | 10 x 6 x 11 x 8 x 10 x 5 | Same | 38 |
| KTH | 30 x 40 x 30 | Same | 190 |

The corresponding publication settings are below. Validation counts refer to samples reserved per class from the initial test pool. The historical driver uses 5 for every dataset. [Paper, Section VI](https://arxiv.org/pdf/1904.06788).

| Dataset | Paper sample shape | Paper training count | Paper validation count | Retained code validation count |
| --- | --- | --- | --- | --- |
| COIL-100 | 8 x 8 x 8 x 8 | 20 | 10 | 5 |
| Weizmann | 4 x 4 x 4 x 4 x 11 | 20 | 5 | 5 |
| Cambridge | 30 x 40 x 30 x 10 | 4 | 1 | 5, which now fails allocation validation |
| UCF-101 | 30 x 40 x 50 | 60 | 10 | 5 |

The paper's lambda search spans 0.1 through 1000 on a logarithmic scale; the code retains `10.^(-2:4)`, spanning 0.01 through 10000. Both use five validation repetitions and ten experiment repetitions. The publication specifies relative singular-value truncation with `tau` in `(0,1]`; code grids can include zero, meaning no truncation. Publication values are provenance, not automatic overrides.

The exact retained arrays are in [dataset_preset.m](../src/data/dataset_preset.m) and [mda_preset.m](../src/data/mda_preset.m). For the TT driver, `tau_list` belongs to LDA; `tau_list2` to TTDA, TTNPE, and BTT; `tau_list3` to TWTTDA, MPS, TWBTT, and ThWBTT; `tau_list4` to ThW. The MDA driver uses its own `tau_list` for both comparisons. Independent iteration preserves every value when grid lengths differ.

The scatter implementation also retains covariance normalization, while paper equations (6)-(7) use sums. For balanced classes this changes the effective weighting of lambda. See [the numerical-change notes](changes.md).

Incomplete old presets now give targeted errors:

- UCF101 has no `tau_list2` for TTDA/TTNPE/BTT.
- Cambridge, MNIST, YaleB, GAIT, and KTH have no three-branch TTDA grid.
- Some original fixed MDA rank tables have incompatible widths or noninteger/out-of-range ranks.
- Weizmann's five-entry all-zero ablation row has been resized to its actual three-mode TT shape; its effect is still zero ablation.

Supply an explicit grid when needed:

```matlab
config = experiment_config('Cambridge');
config.data_dir = 'D:/research-data/mbttda';
config.tau_by_method.THW = [0.2, 0.1]; % Explicit user-selected values, not a recovered paper grid.
config.validation_per_class = 1;
config.repeats = 1;
experiment = run_experiment(config);
```

For COIL3D, a validation size of 2 leaves final test data and fits the training pool. A fixed `lambda` bypasses validation entirely.

## MAT-file format

Each file must contain a numeric variable named `Data`. Its last two axes are **samples per class** and **class**, preceded by physical feature axes. Class labels are generated from these final axes. The special historical YaleB layout is permuted as in the original loader.

Headers of bundled files were inspected without loading them into MATLAB:

| File | Data dimensions |
| --- | --- |
| `CoilData.mat` | 64 x 64 x 72 x 100 |
| `COIL3Ddata.mat` | 64 x 64 x 8 x 9 x 100 |
| `MNIST.mat` | 28 x 28 x 600 x 10 |
| `WeizmannData.mat` | 64 x 44 x 45 x 28 |
| `coil_small.mat` | 16 x 16 x 18 x 20; no original named experiment preset |

External filenames are `handGestures.mat` (Cambridge), `UCF101Data.mat`, `YaleBData.mat`, `GaitData.mat`, and `KTHData.mat`. Their feature-axis product must match the selected tensor shape. Existing presets expect balanced preprocessed arrays, not raw videos. This checkout does not contain the original Cambridge/UCF preprocessing scripts.

The COIL TT loader retains its bit-interleaving permutation. The MDA loader uses its original direct reshaping.

### Scaling

The historical scaling is intentionally visible:

- All splits divide loader output by `config.scale`, default 255.
- The TT loader already divides Cambridge/KTH by 255.
- The MDA loader already divides GAIT/YaleB/Weizmann/Cambridge/KTH by 255.
- Consequently those combinations retain two divisions by 255 under default settings.

Use `scale = 1` for already normalized custom data. Changing scaling in a historical experiment is an explicit configuration change.

### Custom data

```matlab
config = experiment_config('custom');
config.images = X;             % features x observations
config.labels = y;             % one numeric label per column
config.tensor_shape = [2,3,2];
config.train_per_class = 6;
config.scale = 1;
config.tau = 0.2;
config.lambda = 1;
config.repeats = 1;
experiment = run_experiment(config);
```

## Validation and rank selection

Splitting selects a fixed training count independently within each class. When lambda search is enabled, a reserved number of samples is removed from the initial test pool and combined with training data. Repeated leave-s-out folds use only that combined pool. The final model uses the pool, and the final holdout is excluded from all validation folds.

Original training indices, reserved validation indices, final holdout indices, and fold indices are recorded. The code retains the common within-class fold permutation and the last-best-candidate tie rule. Zero-noise runs no longer consume random draws for unused noise arrays, so old unrecorded splits are not reproduced by matching a seed alone.

Automatic MDA ranks in the modern runner use training data only. The compatibility function `load_dataMDA` retains full-data rank estimation for callers requesting its original seven outputs.

## Result format

Each checkpoint stores `record` and run `metadata`; the final `experiment.mat` stores `experiment`, its config, environment, records, and summary table. Both loading paths preserve configuration and metric definitions. Directories with existing MAT files are rejected to avoid overwriting runs. Completed checkpoints remain loadable if a later computation fails.

`record.parameters` contains the method settings, and `record.ranks` records chosen MDA ranks. Sample indices refer to the loaded column order. `train_per_class` is the configured initial count; `metrics.training_samples` reports the actual final training pool, including reserved validation samples when a search ran. Custom input matrices are not copied into metadata; retain them separately for reproducibility. The recorded Git revision identifies the base commit, not uncommitted source changes.

Subspace time includes initialization and, for MDA, training-only rank estimation. Positive lambda values fix the branch objective weighting; nonpositive values retain the legacy TT/BTT automatic weighting calculation inside each branch solve. Dense LDA uses the supplied lambda directly.

Results retain `PreLabel`, `PreErr`, `Storage`, and timing fields. `metrics.storage_ratio` counts factor and training-feature elements. Historical `Storage` values keep their original measurement conventions; they must not be compared directly with element ratios.

| Legacy `Storage` convention | Methods / entry points |
| --- | --- |
| Byte ratio including cell overhead | TTDA and BTT variants, MPS, modern/struct MDA calls |
| Dense byte ratio, equivalent to element ratio | LDA |
| Manifold-dimension estimate plus feature elements, unnormalized | TTNPE |
| Absolute factor and feature element count | Eight-argument `main_CMDA` / `main_DGTDA` compatibility calls |

New results include `metrics.legacy_storage_definition` so these conventions remain inspectable. Use `metrics.storage_ratio` for comparisons across methods.

Plots average available repeats at each method/setting, separating dataset, training size, ablation, and legacy metrics. Historical filename holdout/shape text is retained for grouping; unreliable old `hout` values are not converted into invented sample counts. Their filename tau can be a shared sweep label rather than the actual method threshold. Missing timing fields are reported. Partial checkpoints contain only completed runs; their plots are partial, not full-sweep reproductions.
