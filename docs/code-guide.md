# Reading the implementation

Start with `demo_synthetic`, then `run_experiment`. A useful reading order is data preparation, TT initialization, one-branch training, and finally the two- and three-branch variants.

## Paper-to-code guide

| Paper concept | Code |
| --- | --- |
| Relative singular-value thresholding | `U2Ui_tau`, `svdtrunc2` |
| Class scatter and signed discriminant objective | `const_scat`, `lrnU`, `ObjVal` |
| Orthogonal TT-core updates, Algorithm 1 | `TensNet_Solver`, `core_quadratic`, `Ui_Solver`, `Un_Solver` |
| Two-branch alternation, Algorithm 2 | `twttda` |
| Three-branch alternation, Section IV-B | `thwttda` |
| Contiguous branch partition | `computeK` |
| Feature extraction | `merge_tensor`, `project_branch_features` |
| Classification and storage reporting | `Classfier_KNN`, `result_metrics` |

The reference is [arXiv v2](https://arxiv.org/abs/1904.06788v2). Comments in the solver point to equations (10), (12), and (14).

## Array conventions

Public algorithm inputs use a struct:

```matlab
data.train   % features x training samples
data.test    % features x test samples
data.trLbl   % one training label per column
data.tsLbl   % one test label per column
```

Labels are finite numeric values; they need not start at one. Training data must contain at least two classes, each with the same number of samples, at least two per class. The validator groups training columns by label and preserves test-column order.

A sample's physical dimensions are `parameters.I`. Their product must match the feature count. A TT core has dimensions `[leftRank, physicalDimension, rightRank]`. The final coefficient cell returned by `U2Ui_tau` holds sample features.

Scatter calculations use `[features, samplesPerClass, classCount, projectedCoordinates]`. Explicit sizes preserve singleton feature/rank dimensions. Cyclic mode unfolding follows the original TP Toolbox order and requires no Wavelet Toolbox.

## A branch update

`lrnU` forms the signed scatter matrix, reshapes it into the branch's physical modes, and calls `TensNet_Solver`. Its outputs are:

```matlab
[cores, objectiveHistory, scatterSeconds, searchSeconds] = lrnU(data, cores, parameters);
```

For each core, `core_quadratic` contracts the fixed left and right environments. It builds only that core's effective quadratic, within the current branch. The final core uses a trace objective shared by its output columns.

The two- and three-branch functions retain explicit partial projections and alternating updates. They do not construct a global Kronecker projection. The small dense maps in the tests are independent reference calculations, not production implementations.

## Parameters and compatibility

Use `algorithm_parameters(shape, overrides)` for defaults. The important settings are `tau`, `lambda`, `maxiter`, `maxiterOut`, `error_tot`, `nA`, and `rndI`.

`main_CMDA` and `main_DGTDA` accept both historical argument layouts. `get_data` accepts the struct-returning and four-output layouts. `OWTTDA`, `TT2WDA`, and `mymain_App2` delegate to the maintained variants.

`Classfier_KNN` keeps its original spelling to avoid breaking calls. `thwttda.m` now matches its function name on case-sensitive systems.

Call `setup_mbttda` after opening the checkout; adding only the former `functs` directory is no longer sufficient.
