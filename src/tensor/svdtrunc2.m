function [basis, values, nextValue] = svdtrunc2(matrix, threshold)
    % SVDTRUNC2 Keep singular values >= threshold times the largest singular value.
    % The third output is the largest discarded singular value (zero if none).
    if nargin < 2
        threshold = eps;
    end
    validateattributes(threshold, {'numeric'}, {'scalar', 'finite', '>=', 0, '<=', 1});
    validateattributes(matrix, {'numeric'}, {'2d', 'real', 'finite', 'nonempty'});
    [basis, diagonal] = svd(matrix, 'econ');
    values = diag(diagonal);
    rank = sum(values >= threshold * max(values));
    nextValue = 0;
    if rank < numel(values)
        nextValue = values(rank + 1);
    end
    basis = basis(:, 1:rank);
    values = values(1:rank);
end
