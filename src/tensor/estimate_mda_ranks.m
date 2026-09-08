function ranks = estimate_mda_ranks(training, shape, tau)
    % ESTIMATE_MDA_RANKS Per-mode singular-value thresholding on training columns.
    tensor = reshape(training, [shape, size(training, 2)]);
    ranks = zeros(size(shape));
    for mode = 1:numel(shape)
        [basis, ~] = svdtrunc2(ndim_unfold(tensor, mode), tau);
        ranks(mode) = size(basis, 2);
    end
end
