function features = project_branch_features(samples, projections)
    % PROJECT_BRANCH_FEATURES Contract branch factors without a full Kronecker map.
    % samples: [left dimension, observations, other branch dimensions].
    % Feature order follows cyclic mode-2 unfolding, matching the original code.
    matrices = cellfun(@transpose, projections, 'UniformOutput', false);
    modes = [1, 3:numel(projections) + 1];
    projected = tmprod(samples, matrices, modes);
    features = ndim_unfold(projected, 2)';
end
