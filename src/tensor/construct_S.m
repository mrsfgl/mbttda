function graph = construct_S(samples, ~, neighbors, epsilon)
    % CONSTRUCT_S Row-normalized heat-kernel neighborhood graph used by TTNPE.
    % The original label mask was disabled; this remains an unsupervised graph.
    samples = reshape(samples, [], size(samples, ndims(samples)));
    count = size(samples, 2);
    validateattributes(neighbors, {'numeric'}, {'scalar', 'integer', 'positive', '<', count});
    validateattributes(epsilon, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
    [indices, distances] = knnsearch(samples', samples', 'K', neighbors + 1, 'Distance', 'euclidean');
    graph = zeros(count);
    for i = 1:count
        % Remove the actual self-index, including when duplicate observations tie.
        keep = indices(i, :) ~= i;
        candidates = indices(i, keep);
        distance = distances(i, keep);
        candidates = candidates(1:neighbors);
        distance = distance(1:neighbors);
        graph(i, candidates) = exp(-distance.^2 / epsilon) + 1e-8;
    end
    graph = graph + graph';
    graph = graph ./ sum(graph, 2);
end
