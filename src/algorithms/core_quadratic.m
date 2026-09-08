function quadratic = core_quadratic(scatter, cores, mode)
    % CORE_QUADRATIC Contract fixed TT environments around one core (Eqs. 12-14).
    % This works within a single branch. No global multi-branch projection is built.
    shape = cellfun(@(core) size(core, 2), cores);
    leftSize = prod(shape(1:mode - 1));
    rightSize = prod(shape(mode + 1:end));
    physicalSize = shape(mode);
    leftRank = size(cores{mode}, 1);
    if mode == 1
        left = 1;
    else
        left = reshape(merge_tensor(cores(1:mode - 1), false), leftSize, leftRank);
    end
    if mode == numel(cores)
        % Last core: all output columns share the same quadratic matrix.
        folded = reshape(scatter, [leftSize, physicalSize, leftSize, physicalSize]);
        reduced = tmprod(folded, {left', left'}, [1, 3]);
        quadratic = reshape(reduced, leftRank * physicalSize, []);
    else
        rightRank = size(cores{mode}, 3);
        outputRank = size(cores{end}, 3);
        right = reshape(merge_tensor(cores(mode + 1:end), false), rightRank, rightSize, outputRank);
        folded = reshape(scatter, [leftSize, physicalSize, rightSize, leftSize, physicalSize, rightSize]);
        % Project both left environments once, then contract each output coordinate.
        reduced = tmprod(folded, {left', left'}, [1, 4]);
        localSize = leftRank * physicalSize * rightRank;
        quadratic = zeros(localSize);
        for coordinate = 1:outputRank
            rightSlice = right(:, :, coordinate);
            term = tmprod(reduced, {rightSlice, rightSlice}, [3, 6]);
            quadratic = quadratic + reshape(term, localSize, localSize);
        end
    end
    quadratic = (quadratic + quadratic') / 2;
end
