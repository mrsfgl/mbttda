function result = myLDA(data, p)
    % MYLDA Dense discriminant subspace baseline using the trace-difference objective.
    % Retains the historical reciprocal-eigenvalue rank heuristic. Eigenvectors
    % are explicitly ordered by increasing objective value rather than eig order.
    if ~isfield(p, 'I')
        p.I = size(data.train, 1);
    end
    p = algorithm_parameters(p.I, p);
    data = validate_dataset(data, p.I);
    samples = size(data.train, 2);
    classes = numel(unique(data.trLbl));
    perClass = samples / classes;
    timer = tic;
    classMeans = reshape(mean(reshape(data.train, [], perClass, classes), 2), [], classes);
    centered = data.train - repelem(classMeans, 1, perClass);
    within = centered * centered';
    centeredMeans = classMeans - mean(classMeans, 2);
    between = centeredMeans * centeredMeans';
    scatter = within - p.lambda * between;
    [basis, diagonal] = eig((scatter + scatter') / 2);
    [eigenvalues, order] = sort(diag(diagonal), 'ascend');
    basis = basis(:, order);
    if p.tau == 0
        rank = numel(eigenvalues);
    elseif any(eigenvalues == 0)
        rank = sum(eigenvalues == 0);
    else
        scores = 1 ./ eigenvalues;
        rank = max(1, sum(scores > p.tau * max(scores)));
    end
    basis = basis(:, 1:rank);
    result.time_subspace = toc(timer);
    timer = tic;
    features = basis' * data.train;
    testFeatures = basis' * data.test;
    result.time_embedding = toc(timer);
    timer = tic;
    [result.PreLabel, result.PreErr] = Classfier_KNN(features, data.trLbl, testFeatures, data.tsLbl, 1);
    result.time_classify = toc(timer);
    result.metrics = result_metrics({basis}, features, numel(data.train));
    result.Storage = result.metrics.storage_ratio; % Dense double arrays have no cell overhead.
    result.metrics.legacy_storage_definition = 'Dense factor and feature byte count divided by training bytes; equals element ratio';
end
