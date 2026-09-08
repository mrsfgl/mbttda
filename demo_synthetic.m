function experiment = demo_synthetic()
    % DEMO_SYNTHETIC Small, deterministic example of the three TTDA variants.
    setup_mbttda;
    config = experiment_config('synthetic');
    config.seed = 0;
    config.repeats = 1;
    config.scale = 1;
    config.tau = 0.35;
    config.lambda = 1;
    config.parameters = struct('maxiter', 3, 'maxiterOut', 1);
    config.save_results = false;
    experiment = run_experiment(config);
    disp(experiment.summary(:, {'method', 'accuracy', 'storage_ratio', 'subspace_seconds'}));
end
