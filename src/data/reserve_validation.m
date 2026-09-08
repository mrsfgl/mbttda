function [pooled, allocation] = reserve_validation(data, perClass)
    % RESERVE_VALIDATION Move a reserved subset into the training/validation pool.
    % The remaining final test columns are never passed to hyperparameter search.
    validateattributes(perClass, {'numeric'}, {'scalar', 'integer', 'positive'});
    if ~isfield(data, 'train_indices')
        data.train_indices = 1:size(data.train, 2);
    end
    if ~isfield(data, 'test_indices')
        data.test_indices = size(data.train, 2) + (1:size(data.test, 2));
    end
    classes = unique(data.trLbl);
    trainColumns = [];
    reserved = [];
    finalTest = [];
    for label = classes
        tr = find(data.trLbl == label);
        ts = find(data.tsLbl == label);
        if perClass >= numel(ts) || perClass > numel(tr)
            error('mbttda:InsufficientValidation', 'Each class needs at least %d training and %d test-pool samples.', perClass, perClass + 1);
        end
        trainColumns = [trainColumns, tr]; %#ok<AGROW>
        reserved = [reserved, ts(1:perClass)]; %#ok<AGROW>
        finalTest = [finalTest, ts(perClass + 1:end)]; %#ok<AGROW>
    end
    pooled = data;
    pooled.train = [data.train(:, trainColumns), data.test(:, reserved)];
    pooled.trLbl = [data.trLbl(trainColumns), data.tsLbl(reserved)];
    pooled.train_indices = [data.train_indices(trainColumns), data.test_indices(reserved)];
    [pooled.trLbl, order] = sort(pooled.trLbl);
    pooled.train = pooled.train(:, order);
    pooled.train_indices = pooled.train_indices(order);
    pooled.test = data.test(:, finalTest);
    pooled.tsLbl = data.tsLbl(finalTest);
    pooled.test_indices = data.test_indices(finalTest);
    allocation.original_train = data.train_indices(trainColumns);
    allocation.validation_reserved = data.test_indices(reserved);
    allocation.final_test = pooled.test_indices;
end
