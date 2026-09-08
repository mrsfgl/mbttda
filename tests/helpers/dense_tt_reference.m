function matrix = dense_tt_reference(cores)
    % DENSE_TT_REFERENCE Enumerate physical entries; independent of merge_tensor.
    shape = cellfun(@(core) size(core, 2), cores);
    matrix = zeros(prod(shape), size(cores{end}, 3));
    for entry = 1:prod(shape)
        indices = cell(1, numel(shape));
        [indices{:}] = ind2sub(shape, entry);
        value = 1;
        for mode = 1:numel(shape)
            slice = reshape(cores{mode}(:, indices{mode}, :), size(cores{mode}, 1), size(cores{mode}, 3));
            value = value * slice;
        end
        matrix(entry, :) = value;
    end
end
