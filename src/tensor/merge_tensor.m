function tensor = merge_tensor(cores, squeezeBoundaries)
    % MERGE_TENSOR Contract neighboring TT bonds, retaining physical mode order.
    % Without squeezing: [left rank, I1, ..., In, right rank].
    if nargin < 2
        squeezeBoundaries = true;
    end
    if isempty(cores)
        error('mbttda:InvalidCores', 'At least one TT core is required.');
    end
    tensor = cores{1};
    shape = [size(tensor, 1), size(tensor, 2), size(tensor, 3)];
    for i = 2:numel(cores)
        core = cores{i};
        if shape(end) ~= size(core, 1)
            error('mbttda:InvalidCores', 'Adjacent TT ranks must agree.');
        end
        tensor = reshape(tensor, [], shape(end)) * reshape(core, size(core, 1), []);
        shape = [shape(1:end - 1), size(core, 2), size(core, 3)]; %#ok<AGROW>
        tensor = reshape(tensor, shape);
    end
    if squeezeBoundaries
        tensor = wwq_squeeze(tensor);
    end
end
