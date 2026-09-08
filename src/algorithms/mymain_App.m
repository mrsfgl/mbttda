function result = mymain_App(train, trLbl, test, tsLbl, shape, tau, options, neighbors)
    % MYMAIN_APP Historical separate-array interface for dense LDA.
    p = algorithm_parameters(shape, options);
    p.tau = tau;
    if isstruct(neighbors)
        neighbors = neighbors.K;
    end
    if ~isequal(neighbors, 1)
        error('mbttda:InvalidConfig', 'LDA uses 1-NN.');
    end
    data = struct('train', train, 'trLbl', trLbl, 'test', test, 'tsLbl', tsLbl);
    result = myLDA(data, p);
    result.LDAPreLabel = result.PreLabel;
    result.LDAPreErr = result.PreErr;
    result.LDAStorage = result.Storage;
    result.LDAtime_subspace = result.time_subspace;
    result.LDAtime_embedding = result.time_embedding;
    result.LDAtime_classify = result.time_classify;
end
