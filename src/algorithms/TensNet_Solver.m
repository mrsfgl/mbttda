function [cores, objective] = TensNet_Solver(cores, scatter, p, varargin)
    % TENSNET_SOLVER Alternating orthogonal TT-core updates (paper Algorithm 1).
    % Core-local quadratics are computed by contracting fixed left/right factors.
    if isempty(varargin)
        opts = optimizer_options();
    else
        opts = varargin{1};
    end
    objective = ObjVal(scatter, cores);
    for iteration = 1:p.maxiter
        previousLast = cores{end};
        for mode = 1:numel(cores)
            quadratic = core_quadratic(scatter, cores, mode);
            coreShape = [size(cores{mode}, 1), size(cores{mode}, 2), size(cores{mode}, 3)];
            if mode == numel(cores)
                [cores{mode}, ~] = Un_Solver(quadratic, coreShape, opts);
            else
                [cores{mode}, ~] = Ui_Solver(quadratic, coreShape, opts);
            end
            if p.display
                objective(end + 1) = ObjVal(scatter, cores); %#ok<AGROW>
                fprintf('Sweep %d, core %d, objective %.8g\n', iteration, mode, objective(end));
            end
        end
        % A zero previous factor cannot cause NaN/Inf in the stopping criterion.
        change = norm(cores{end}(:) - previousLast(:)) / max(norm(previousLast(:)), eps);
        if ~p.display
            objective(end + 1) = ObjVal(scatter, cores);
        end %#ok<AGROW>
        if change <= p.error_tot
            break
        end
    end
end
