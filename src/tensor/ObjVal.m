function value = ObjVal(scatter, cores, modeCount)
    % OBJVAL tr(U'*(Sw-lambda*Sb)*U), paper Eq. (10), for the current TT subspace.
    if nargin < 3
        modeCount = numel(cores);
    end
    if modeCount ~= numel(cores)
        error('mbttda:InvalidCores', 'Core count does not match tensor order.');
    end
    features = prod(cellfun(@(core) size(core, 2), cores));
    projection = reshape(merge_tensor(cores, false), features, []);
    value = trace(projection' * reshape(scatter, features, features) * projection);
end
