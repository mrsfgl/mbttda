function result = run_method(method, data, parameters, ranks)
    % RUN_METHOD One dispatch point for the existing comparison implementations.
    switch upper(method)
        case 'TTDA'
            result = ttda(data, parameters);
        case 'TWTTDA'
            result = twttda(data, parameters);
        case 'THW'
            result = thwttda(data, parameters);
        case 'LDA'
            result = myLDA(data, parameters);
        case 'MPS'
            result = mps(data, parameters);
        case 'TTNPE'
            result = ttnpe_classifier(data, parameters);
        case 'BTT'
            result = btt(data, parameters);
        case 'TWBTT'
            result = tw_btt(data, parameters);
        case 'THWBTT'
            result = thw_btt(data, parameters);
        case {'CMDA', 'DGTDA'}
            timer = tic;
            if isempty(ranks)
                ranks = estimate_mda_ranks(data.train, parameters.I, parameters.tau);
            end
            initializationSeconds = toc(timer);
            result = mda_classifier(upper(method), data, parameters.I, ranks, 1);
            result.time_subspace = result.time_subspace + initializationSeconds;
        otherwise
            error('mbttda:UnknownMethod', 'Unknown method: %s', method);
    end
end
