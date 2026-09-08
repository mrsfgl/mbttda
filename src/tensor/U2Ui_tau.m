function cores = U2Ui_tau(tensor, tau, randomInit, direction)
    % U2UI_TAU Sequential TT-SVD with relative singular-value thresholding.
    % Final mode contains samples; the final cell holds their feature coefficients.
    % tau=0 retains every singular value (including zero), matching the old preset.
    if nargin < 3
        randomInit = struct('flag', false);
    end
    if nargin < 4
        direction = 1;
    end
    if direction ~= 1
        error('mbttda:UnsupportedLayout', 'The undocumented reverse layout is not supported. Supply modes in forward order.');
    end
    validateattributes(tensor, {'numeric'}, {'real', 'finite', 'nonempty'});
    validateattributes(tau, {'numeric'}, {'scalar', 'finite', '>=', 0, '<=', 1});
    shape = size(tensor);
    cores = cell(1, numel(shape));
    remainder = double(tensor);
    leftRank = 1;
    for mode = 1:numel(shape) - 1
        matrix = reshape(remainder, leftRank * shape(mode), []);
        if randomInit.flag
            if ~isfield(randomInit, 'rank') || numel(randomInit.rank) < numel(shape) - 1
                error('mbttda:InvalidRank', 'Random initialization needs one rank per physical mode.');
            end
            rank = randomInit.rank(mode);
            validateattributes(rank, {'numeric'}, {'scalar', 'integer', 'positive', '<=', size(matrix, 1)});
            basis = orth(rand(size(matrix, 1), rank));
            remainder = basis' * matrix;
        else
            [basis, singularValues, rightVectors] = svd(matrix, 'econ');
            values = diag(singularValues);
            rank = sum(values >= tau * max(values));
            basis = basis(:, 1:rank);
            remainder = singularValues(1:rank, 1:rank) * rightVectors(:, 1:rank)';
        end
        cores{mode} = reshape(basis, leftRank, shape(mode), rank);
        leftRank = rank;
    end
    cores{end} = remainder;
end
