function cuts = computeK(shape, branches)
    % COMPUTEK Balance contiguous groups of physical modes (paper Section IV).
    % Minimize the L1 distance of branch dimensions from the geometric mean.
    % Ties retain the first lexicographic partition, as in the original code.
    if iscell(shape)
        shape = cell2mat(shape);
    end
    validateattributes(shape, {'numeric'}, {'vector', 'integer', 'positive', 'finite'});
    validateattributes(branches, {'numeric'}, {'scalar', 'integer', 'positive'});
    if branches > numel(shape)
        error('mbttda:InvalidBranches', 'The number of branches cannot exceed the number of modes.');
    end
    if branches == 1
        cuts = [];
        return
    end
    candidates = nchoosek(1:numel(shape) - 1, branches - 1);
    target = prod(shape)^(1 / branches);
    cost = zeros(size(candidates, 1), 1);
    for i = 1:size(candidates, 1)
        edges = [0, candidates(i, :), numel(shape)];
        dimensions = arrayfun(@(b) prod(shape(edges(b) + 1:edges(b + 1))), 1:branches);
        cost(i) = sum(abs(dimensions - target));
    end
    [~, best] = min(cost);
    cuts = candidates(best, :);
end
