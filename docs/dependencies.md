# Dependencies

## Core example and tests

The initial validation target is **MATLAB R2024b** with **Statistics and Machine Learning Toolbox** (`fitcknn`; TTNPE also uses `knnsearch`). Earlier releases and Octave have not been validated.

`setup_mbttda` adds only the required bundled directories and this project's source. It changes the current MATLAB session's path, does not persist it with `savepath`, and does not change the working directory.

| Bundled component | Used for |
| --- | --- |
| FOptM | Orthogonality-constrained optimization |
| Tensorlab | Tensor/matrix products and matricization |
| TP Toolbox array helpers | Historical tensor utilities |
| tensor_classification | CMDA and DGTDA |

The project has its own cyclic `ndim_unfold`, matching the old convention while removing the `wshift` dependency. The core example does not require Wavelet, Image Processing, Parallel Computing, or Optimization Toolbox.

## Optional TTNPE comparison

Obtain the original [TTNPE repository](https://github.com/wangwenqi1990/TTNPE) and add its path:

```matlab
setup_mbttda({'D:/toolboxes/TTNPE'});
check_dependencies({'TTNPE'});
```

The approximation solver needs `TenConPro`, `L`, `R`, `L_inv`, `LSQ_Unitary_L`, and `LSQ_Unitary_R`. These helpers are not bundled here. The dependency check verifies symbols; compatibility with a particular checkout still requires the optional smoke test.

The historical `main_TNPE` prototype is separate from the TTNPE comparison. It additionally requires upstream `Tucker_U2Y`, `H_Generator`, `diffU`, and `Dim_Tucker`; it reports missing helpers explicitly.

## Optional block-TT comparisons

Install:

1. [TT-Toolbox](https://github.com/oseledets/TT-Toolbox), providing `tt_matrix` and its `core2cell` method.
2. [TTeMPS 1.1](https://www.epfl.ch/labs/anchp/?p=353), providing `TTeMPS_op` and `amen_eigenvalue`.
3. [Knyazev's LOBPCG implementation](https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m), required by TTeMPS eigenvalue routines.

```matlab
setup_mbttda({'D:/toolboxes/TT-Toolbox', ...
              'D:/toolboxes/TTeMPS_1.1', ...
              'D:/toolboxes/lobpcg'});
check_dependencies({'BTT','TWBTT','ThWBTT'});
```

The adapter follows the TTeMPS 1.1 `[X,C,...] = amen_eigenvalue(...)` interface. It combines the shared factors from `X.U` with every last-core entry in `C`. This interface was inspected in the [TTeMPS copy maintained in Manopt](https://github.com/NicolasBoumal/manopt/tree/master/manopt/manifolds/ttfixedrank/TTeMPS_1.1); integration has not been executed.

No optional toolkit is installed, downloaded, or replaced by `setup_mbttda`. Keep its license/attribution files and record the exact revision you install when validating a benchmark.

## Checking availability

```matlab
missing = check_dependencies({'TTDA','MPS'}, false);
disp(missing);
```

With the second argument omitted, missing functions cause an error. The standard test suite treats missing core dependencies as failures and missing optional dependencies as explicit assumption-based skips.

The prepared workflow uses the official [Setup MATLAB](https://github.com/matlab-actions/setup-matlab) and [Run MATLAB Tests](https://github.com/matlab-actions/run-tests) actions. Optional external algorithms remain skipped until those dependencies are installed on a runner.
