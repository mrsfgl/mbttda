function data = validate_dataset(data, shape)
    % VALIDATE_DATASET Enforce balanced, class-contiguous training layout.
    % Samples are columns. Test columns retain their original input order.
    if ~isstruct(data) || ~all(isfield(data, {'train', 'test', 'trLbl', 'tsLbl'}))
        error('mbttda:InvalidData', 'Data requires train, test, trLbl, and tsLbl.');
    end
    data.trLbl = reshape(data.trLbl, 1, []);
    data.tsLbl = reshape(data.tsLbl, 1, []);
    for name = {'train', 'test'}
        validateattributes(data.(name{1}), {'numeric'}, {'2d', 'real', 'finite', 'nonempty'});
        if size(data.(name{1}), 1) ~= prod(shape)
            error('mbttda:InvalidShape', 'Feature count must equal prod(tensor_shape).');
        end
        data.(name{1}) = double(data.(name{1}));
    end
    validateattributes(data.trLbl, {'numeric'}, {'real', 'finite'});
    validateattributes(data.tsLbl, {'numeric'}, {'real', 'finite'});
    if numel(data.trLbl) ~= size(data.train, 2) || numel(data.tsLbl) ~= size(data.test, 2)
        error('mbttda:InvalidLabels', 'Each sample column must have one label.');
    end
    classes = unique(data.trLbl);
    counts = arrayfun(@(label) sum(data.trLbl == label), classes);
    if numel(classes) < 2 || any(counts ~= counts(1)) || counts(1) < 2
        error('mbttda:UnbalancedData', 'Training needs at least two classes with equal counts of at least two samples.');
    end
    if ~all(ismember(data.tsLbl, classes))
        error('mbttda:InvalidLabels', 'Test labels must occur in training.');
    end
    [data.trLbl, order] = sort(data.trLbl);
    data.train = data.train(:, order);
    if isfield(data, 'train_indices')
        data.train_indices = data.train_indices(order);
    end
end
