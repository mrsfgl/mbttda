function fold = validation_fold(pool, trainingPerClass)
    % VALIDATION_FOLD Class-balanced leave-s-out split of the reserved pool only.
    classes = unique(pool.trLbl);
    count = sum(pool.trLbl == classes(1));
    % Preserve the original common within-class permutation across classes.
    chosen = randperm(count, trainingPerClass);
    heldOut = setdiff(1:count, chosen, 'stable');
    trainColumns = [];
    testColumns = [];
    for label = classes
        columns = find(pool.trLbl == label);
        if numel(columns) ~= count
            error('mbttda:UnbalancedData', 'Validation pool must be class balanced.');
        end
        trainColumns = [trainColumns, columns(chosen)]; %#ok<AGROW>
        testColumns = [testColumns, columns(heldOut)]; %#ok<AGROW>
    end
    fold = struct('train', pool.train(:, trainColumns), 'test', pool.train(:, testColumns), ...
                  'trLbl', pool.trLbl(trainColumns), 'tsLbl', pool.trLbl(testColumns), ...
                  'train_indices', pool.train_indices(trainColumns), 'test_indices', pool.train_indices(testColumns));
end
