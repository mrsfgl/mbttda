function result = ttnpe_classifier(data, p)
    % TTNPE_CLASSIFIER Classification wrapper for the existing approximation solver.
    % Optional upstream TTNPE helpers are required; see docs/dependencies.md.
    data = validate_dataset(data, p.I);
    check_dependencies({'TTNPE'});
    p.disp = p.display;
    if ~isfield(p, 'Graph')
        p.Graph = struct('K', size(data.train, 2) - 1, 'epsilon', 1);
    end
    timer = tic;
    cores = U2Ui_tau(reshape(data.train, [p.I, size(data.train, 2)]), p.tau);
    cores = cores(1:end - 1);
    graph = construct_S(data.train, data.trLbl, p.Graph.K, p.Graph.epsilon);
    residual = data.train * (eye(size(data.train, 2)) - graph');
    scatter = reshape(residual * residual', [p.I, p.I]);
    [cores, ~, ~, ~, ~] = Apro_Solver(cores, scatter, p, optimizer_options());
    projection = reshape(merge_tensor(cores, false), prod(p.I), []);
    result.time_subspace = toc(timer);
    timer = tic;
    features = projection' * data.train;
    testFeatures = projection' * data.test;
    result.time_embedding = toc(timer);
    timer = tic;
    [result.PreLabel, result.PreErr] = Classfier_KNN(features, data.trLbl, testFeatures, data.tsLbl, 1);
    result.time_classify = toc(timer);
    result.metrics = result_metrics(cores, features, numel(data.train));
    result.Storage = Dim_TT(cores) + numel(features);
    result.metrics.legacy_storage_definition = 'Original TT manifold-dimension estimate plus training feature elements (absolute count)';
end
