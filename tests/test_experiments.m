function tests = test_experiments
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.addTeardown(@path, path);
    testCase.addTeardown(@rng, rng);
    addpath(fullfile(fileparts(mfilename('fullpath')), 'helpers'));
    enter_test_environment();
end

function testRunnerSavingAndStateRestoration(testCase)
    folder = tempname;
    cleanup = onCleanup(@() removeTestFolder(folder)); %#ok<NASGU>
    config = experiment_config('synthetic');
    config.methods = {'MPS'};
    config.repeats = 1;
    config.scale = 1;
    config.tau = [0.3, 0.5];
    config.lambda = 1;
    config.output_dir = folder;
    beforeRng = rng;
    beforeFolder = pwd;
    result = run_experiment(config);
    testCase.verifyEqual(rng, beforeRng);
    testCase.verifyEqual(pwd, beforeFolder);
    testCase.verifyEqual(numel(result.records), 2);
    loaded = load_experiment(folder);
    testCase.verifyEqual(loaded.summary, result.summary);
    testCase.verifyEqual(loaded.config, result.config);
    testCase.verifyEqual(loaded.environment, result.environment);
    testCase.verifyEqual(loaded.metric_notes, result.metric_notes);
    checkpoints = dir(fullfile(folder, 'run_*.mat'));
    checkpoint = load_experiment(fullfile(folder, checkpoints(1).name));
    testCase.verifyEqual(checkpoint.config, result.config);
    testCase.verifyEqual(checkpoint.records.split, result.records(1).split);
    testCase.verifyEmpty(intersect(result.records(1).split.train, result.records(1).split.test));
    testCase.verifyError(@() run_experiment(config), 'mbttda:ExistingResults');
    figures = plot_experiment(loaded, 'off');
    testCase.verifyEqual(numel(figures), 1);
    close(figures);
end

function testLegacyResultsAndIncompleteMetrics(testCase)
    folder = tempname;
    mkdir(folder);
    cleanup = onCleanup(@() removeTestFolder(folder)); %#ok<NASGU>
    TTDA = struct('PreErr', 0.25, 'Storage', 0.4, 'time_subspace', 2); %#ok<NASGU>
    save(fullfile(folder, 'Result_repeat1_tau0.2_ablation1.mat'), 'TTDA');
    result = load_experiment(folder);
    testCase.verifyEqual(result.summary.accuracy, 0.75);
    testCase.verifyTrue(isnan(result.summary.storage_ratio));
    testCase.verifyEqual(result.summary.legacy_storage, 0.4);
    before = pwd;
    figures = plot_experiment(result, 'off');
    close(figures);
    testCase.verifyEqual(pwd, before);
    save(fullfile(folder, 'Result_repeat1_tau0.2_holdout0.5_ablation1_shape2  3.mat'), 'TTDA');
    save(fullfile(folder, 'Result_repeat1_tau0.2_holdout0.5_ablation1_shape3  2.mat'), 'TTDA');
    grouped = load_experiment(folder);
    testCase.verifyEqual(numel(unique(grouped.summary.legacy_condition)), 3);
    figures = plot_experiment(grouped, 'off');
    testCase.verifyEqual(numel(figures), 3);
    close(figures);
end

function testIndependentGridsAndNeighbors(testCase)
    original = experiment_config('COIL');
    preset = dataset_preset('COIL');
    testCase.verifyEqual(method_tau_grid('TTNPE', original, preset), preset.tau_list2);
    config = experiment_config('synthetic');
    config.methods = {'MPS', 'LDA'};
    config.repeats = 1;
    config.tau_by_method = struct('MPS', [0.3, 0.5], 'LDA', 0.4);
    config.lambda = 1;
    config.save_results = false;
    result = run_experiment(config);
    testCase.verifyEqual(numel(result.records), 3);
    [data, p] = tiny_fixture();
    testCase.verifyError(@() main_App(data.train, data.trLbl, data.test, data.tsLbl, ...
                                      p.I, p.tau, struct('K', 2, 'epsilon', 1), p, struct('K', 2)), 'mbttda:InvalidConfig');
end

function testLambdaSearchHoldoutAndResultShape(testCase)
    [data, p] = tiny_fixture();
    options = struct('candidates', [0.1, 1], 'repeats', 1, 'maxiter', 1, 'methods', 1);
    [lambda, accuracy, pool, validation] = optLambda(data, p.I, 6, 2, options);
    testCase.verifySize(accuracy, [1, 2, 3]);
    testCase.verifyTrue(ismember(lambda(1), options.candidates));
    testCase.verifyEmpty(intersect(validation.folds{1}.train, pool.test_indices));
    testCase.verifyEmpty(intersect(validation.folds{1}.validation, pool.test_indices));
    testCase.verifyEqual(sort(pool.test_indices), sort(validation.allocation.final_test));
end

function testMdaEntryPointCompatibility(testCase)
    [data, p] = tiny_fixture();
    ranks = ones(size(p.I));
    first = main_CMDA(data, p.I, ranks, 1, false);
    second = main_CMDA(data.train, data.trLbl, data.test, data.tsLbl, p.I, ranks, struct('K', 1), false);
    testCase.verifyEqual(first.PreLabel, second.PreLabel);
    testCase.verifyEqual(first.metrics.storage_ratio, second.metrics.storage_ratio);
    testCase.verifyEqual(second.Storage, second.metrics.factor_elements + second.metrics.feature_elements);
    testCase.verifyNotEqual(first.metrics.legacy_storage_definition, second.metrics.legacy_storage_definition);
end

function testInvalidConfig(testCase)
    config = experiment_config('synthetic');
    config.misspelled = true;
    testCase.verifyError(@() run_experiment(config), 'mbttda:InvalidConfig');
    testCase.verifyError(@() check_dependencies({'UNSUPPORTED'}, false), 'mbttda:UnknownMethod');
end

function testLdaUsesClassMeansAndTtdaAcceptsUnequalTestSize(testCase)
    data = struct('train', [-3, -3, -3, 3, 3, 3; -1, 0, 1, -1, 0, 1], ...
                  'trLbl', [1, 1, 1, 2, 2, 2], 'test', [-3, 3; 10, -10], 'tsLbl', [1, 2]);
    result = myLDA(data, struct('tau', 0.2, 'lambda', 1));
    testCase.verifyEqual(result.PreLabel, [1; 2]);
    testCase.verifyEqual(result.PreErr, 0);
    [data, p] = tiny_fixture();
    data.test = data.test(:, [1, 2, 7]);
    data.tsLbl = data.tsLbl([1, 2, 7]);
    p.I = [4, 2, 8];
    p.nA = [0, 0, 0];
    p.maxiter = 1;
    result = ttda(data, p);
    testCase.verifySize(result.PreLabel{end}, [3, 1]);
    testCase.verifyTrue(isfinite(result.PreErr));
end

function removeTestFolder(folder)
    if isfolder(folder)
        rmdir(folder, 's');
    end
end
