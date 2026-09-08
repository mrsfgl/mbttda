function opts = optimizer_options()
    % OPTIMIZER_OPTIONS Shared orthogonality-constrained solver settings.
    opts = struct('record', 0, 'mxitr', 1000, 'gtol', 1e-5, 'xtol', 1e-5, 'ftol', 1e-8, 'tau', 1e-3);
end
