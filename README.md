# mbttda
## Multi Branch Tensor Train Disriminant Analysis

Code for our paper with the same title published in IEEE TIP 

[IEEE link](https://ieeexplore.ieee.org/abstract/document/9585029)

[arxiv link](https://arxiv.org/abs/1904.06788)

## How to use:
`mydemo.m` is a script that runs most of the experimental settings. It uses:

1. A data loading utility `load_data.m`, with a wide variety of parameter options for algorithms.
2. A cross validation function `optLambda.m` that finds the best $\lambda$ value for various number of 'branches'.
3. Functions for each algorithm compared in the paper. See `mdademo.m` for Tucker based algorithms.
4. A visualization utility, `figPlot.m`, that can plot accuracy and computation time w.r.t. storage complexity.

The algorithm saves the results for each set of parameters, and random initialization of experiments in a new folder. See line 80 in `mydemo.m` for some details regarding file name convention. The visualization utility reads from these files to plot.


## Acknowledgement:
This toolbox was created with heavy utilization of [TTNPE](https://github.com/wangwenqi1990/TTNPE) toolbox, and other third-party toolboxes provided in the folder `third-party`.
