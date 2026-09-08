% MDADEMO Original Weizmann Tucker comparisons, with training-only rank selection.
setup_mbttda;
config = experiment_config('Weizmann');
config.preset_family = 'mda';
config.methods = {'CMDA', 'DGTDA'};
config.plot = true;
experiment = run_experiment(config);
