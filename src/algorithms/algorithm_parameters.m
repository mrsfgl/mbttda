function p = algorithm_parameters(shape, overrides)
    % ALGORITHM_PARAMETERS Defaults shared by the original TT experiments.
    validateattributes(shape, {'numeric'}, {'row', 'integer', 'positive', 'finite'});
    p = struct('I', shape, 'tau', 0.2, 'lambda', 1, 'maxiter', 200, ...
               'maxiterOut', 1, 'error_tot', 0.1, 'display', false, 'display2', false, ...
               'nA', zeros(size(shape)), 'rndI', struct('flag', false));
    if nargin > 1
        names = fieldnames(overrides);
        for i = 1:numel(names)
            p.(names{i}) = overrides.(names{i});
        end
    end
    validateattributes(p.I, {'numeric'}, {'row', 'integer', 'positive', 'finite'});
    validateattributes(p.tau, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<=', 1});
    validateattributes(p.lambda, {'numeric'}, {'scalar', 'real', 'finite'});
    validateattributes(p.maxiter, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(p.maxiterOut, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(p.error_tot, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'});
    validateattributes(p.nA, {'numeric'}, {'row', 'integer', 'nonnegative', 'numel', numel(p.I)});
end
