% MYDEMO Original COIL comparison collection, now using the shared runner.
setup_mbttda;
config = experiment_config('COIL');
config.methods = {'TTDA', 'TWTTDA', 'ThW', 'LDA', 'MPS', 'TTNPE', 'BTT', 'TWBTT', 'ThWBTT'};
config.plot = true;
experiment = run_experiment(config);
