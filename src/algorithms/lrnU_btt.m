function [cores, seconds] = lrnU_btt(data, cores, parameters)
    % LRNU_BTT Existing EVAMEn baseline, using TT-Toolbox and TTeMPS 1.1.
    % TTeMPS returns shared cores X and a separate last core C for EACH eigenvector.
    % Recombine all C entries; retaining X alone loses every vector except the first.
    timer = tic;
    shape = cellfun(@(core) size(core, 2), cores);
    outputRank = size(cores{end}, 3);
    [within, between] = const_scat(data);
    lambda = parameters.lambda;
    if lambda <= 0
        lambda = svds(within \ between, 1);
    end
    scatter = within - lambda * between;
    if numel(cores) == 1
        % A single-mode branch is a small ordinary symmetric eigenproblem.
        [basis, diagonal] = eig((scatter + scatter') / 2);
        [~, order] = sort(diag(diagonal), 'ascend');
        cores = {reshape(basis(:, order(1:outputRank)), 1, shape(1), outputRank)};
    else
        check_dependencies({'BTT'});
        operator = TTeMPS_op(core2cell(tt_matrix(reshape(scatter, [shape, shape]), sqrt(parameters.tau))));
        initialRanks = [cellfun(@(core) size(core, 1), cores), 1];
        options = struct('maxiter', 3, 'maxrank', 8, 'tol', parameters.tau, 'precInner', false, 'verbose', 0);
        [solution, lastCores] = amen_eigenvalue(operator, [], outputRank, initialRanks, options);
        cores = solution.U;
        cores{end} = cat(3, lastCores{:});
        cores = reshape(cores, 1, []);
    end
    seconds = toc(timer);
end
