function matrix = ndim_unfold(tensor, mode)
    % NDIM_UNFOLD Cyclic mode unfolding, matching the original TP Toolbox convention.
    % Uses an explicit permutation in place of the Wavelet Toolbox wshift call.
    validateattributes(mode, {'numeric'}, {'scalar', 'positive', 'integer'});
    order = max(ndims(tensor), mode);
    matrix = reshape(permute(tensor, [mode:order, 1:mode - 1]), size(tensor, mode), []);
end
