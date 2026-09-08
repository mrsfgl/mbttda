function result = mda_classifier(method, data, shape, ranks, neighbors)
    % MDA_CLASSIFIER Shared setup, projection, and metrics for CMDA and DGTDA.
    % The bundled implementations use [I1,...,In,samples per class,classes].
    data = validate_dataset(data, shape);
    validateattributes(ranks, {'numeric'}, {'row', 'integer', 'positive'});
    if numel(ranks) > numel(shape) || any(ranks > shape(1:numel(ranks)))
        error('mbttda:InvalidRank', 'MDA ranks cannot exceed their physical mode dimensions.');
    end
    classes = unique(data.trLbl);
    samples = size(data.train, 2) / numel(classes);
    training = reshape(data.train, [shape, samples, numel(classes)]);
    testing = reshape(data.test, [shape, size(data.test, 2)]);
    timer = tic;
    switch method
        case 'CMDA'
            factors = CMDA(training, [], ranks);
        case 'DGTDA'
            factors = DGTDA(training, ranks);
        otherwise
            error('mbttda:UnknownMethod', 'Expected CMDA or DGTDA.');
    end
    result.time_subspace = toc(timer);
    timer = tic;
    matrices = cellfun(@transpose, factors, 'UniformOutput', false);
    features = tmprod(training, matrices, 1:numel(factors));
    testFeatures = tmprod(testing, matrices, 1:numel(factors));
    features = reshape(features, [], size(data.train, 2));
    testFeatures = reshape(testFeatures, [], size(data.test, 2));
    result.time_embedding = toc(timer);
    timer = tic;
    [result.PreLabel, result.PreErr] = Classfier_KNN(features, data.trLbl, testFeatures, data.tsLbl, neighbors);
    result.time_classify = toc(timer);
    result.metrics = result_metrics(factors, features, numel(data.train));
    factorInfo = whos('factors');
    featureInfo = whos('features');
    result.Storage = (factorInfo.bytes + featureInfo.bytes) / (8 * numel(data.train));
    result.Ui = factors;
    result.ranks = ranks;
end
