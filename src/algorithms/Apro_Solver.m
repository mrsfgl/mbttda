function [cores, target, objective, approximationSeconds, eigenSeconds] = Apro_Solver(cores, scatter, p, options)
    % APRO_SOLVER Approximate the lowest graph-scatter eigenspace with TT factors.
    % This is the existing TTNPE alternating least-squares comparison, not TTDA's
    % discriminant solver. External TTNPE helpers supply orthogonal LSQ updates.
    shape = cellfun(@(core) size(core, 2), cores);
    ranks = [1, cellfun(@(core) size(core, 3), cores)];
    modeCount = numel(cores);
    outputRank = ranks(end);
    timer = tic;
    target = smallest_eigenvectors(reshape(scatter, prod(shape), prod(shape)), outputRank);
    eigenSeconds = toc(timer);
    objective = ObjVal(scatter, cores);
    timer = tic;
    if modeCount == 1
        cores = {reshape(target, 1, shape(1), outputRank)};
        objective(end + 1) = ObjVal(scatter, cores);
    else
        for iteration = 1:p.maxiter
            previousLast = cores{end};
            for mode = 1:modeCount
                % Fixed environments turn the tensor fit into a small matrix LSQ.
                if mode == 1
                    right = R(TenConPro(cores(2:end)));
                    desired = reshape(target, shape(1), []);
                    [basis, ~] = LSQ_Unitary_L(right, desired, options);
                elseif mode == modeCount
                    left = L(TenConPro(cores(1:end - 1)));
                    design = kron(eye(shape(mode)), left);
                    [basis, ~] = LSQ_Unitary_R(design, target, options);
                else
                    left = L(TenConPro(cores(1:mode - 1)));
                    right = R(TenConPro(cores(mode + 1:end)));
                    desired = reshape(target, prod(shape(1:mode)), []);
                    design = kron(eye(shape(mode)), left);
                    initial = eye(ranks(mode) * shape(mode), ranks(mode + 1));
                    [basis, ~] = OptStiefelGBB(initial, @AXBC, options, design, right, desired);
                end
                cores{mode} = L_inv(basis, ranks(mode), shape(mode), ranks(mode + 1));
                objective(end + 1) = ObjVal(scatter, cores); %#ok<AGROW>
            end
            change = norm(cores{end}(:) - previousLast(:)) / max(norm(previousLast(:)), eps);
            if isfield(p, 'disp') && p.disp
                fprintf('TTNPE sweep %d: factor change %.5g, objective %.8g\n', iteration, change, objective(end));
            end
            if change <= p.error_tot
                break
            end
        end
    end
    approximationSeconds = toc(timer);
end
