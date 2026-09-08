function [cores, objective, scatterSeconds, searchSeconds] = lrnU(data, cores, p, varargin)
    % LRNU Learn a branch: covariance scatter, then orthogonal TT optimization.
    % Output order is explicit: factors, objective history, scatter time, solver time.
    % Paper Eq. (5), with the repository's retained covariance normalization.
    if isempty(varargin)
        opts = optimizer_options();
    else
        opts = varargin{1};
    end
    shape = cellfun(@(core) size(core, 2), cores);
    timer = tic;
    [within, between] = const_scat(data);
    lambda = p.lambda;
    if lambda <= 0
        lambda = svds(within \ between, 1);
    end
    scatter = reshape(within - lambda * between, [shape, shape]);
    scatterSeconds = toc(timer);
    timer = tic;
    [cores, objective] = TensNet_Solver(cores, scatter, p, opts);
    searchSeconds = toc(timer);
end
