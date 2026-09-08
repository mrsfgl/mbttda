function result = legacy_tt_call(method, train, trainLabels, test, testLabels, shape, tau, options, neighbors)
    % LEGACY_TT_CALL Compatibility adapter for the older separate-array interfaces.
    if isfield(options, 'maxiterL')
        options.maxiterOut = options.maxiterL;
    end
    p = algorithm_parameters(shape, options);
    p.tau = tau;
    if isstruct(neighbors)
        neighbors = neighbors.K;
    end
    if ~isequal(neighbors, 1)
        error('mbttda:InvalidConfig', 'The modern TT experiment wrappers use the paper''s 1-NN classifier.');
    end
    data = struct('train', train, 'trLbl', trainLabels, 'test', test, 'tsLbl', testLabels);
    result = feval(method, data, p);
end
