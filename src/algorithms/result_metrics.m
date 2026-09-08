function metrics = result_metrics(factors, features, trainingElements)
    % RESULT_METRICS Paper Section VI element counts, excluding container overhead.
    metrics.factor_elements = sum(cellfun(@numel, factors));
    metrics.feature_elements = numel(features);
    metrics.training_elements = trainingElements;
    metrics.training_samples = size(features, 2);
    metrics.feature_dimension = size(features, 1);
    metrics.storage_ratio = (metrics.factor_elements + numel(features)) / trainingElements;
    metrics.storage_definition = '(factor elements + training feature elements) / training data elements';
    metrics.legacy_storage_definition = 'Original measurement; usually whos bytes including cell overhead';
end
