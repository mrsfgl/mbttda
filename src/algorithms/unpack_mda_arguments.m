function [data, shape, ranks, neighbors] = unpack_mda_arguments(varargin)
    % UNPACK_MDA_ARGUMENTS Normalize both historical MDA calling conventions.
    if numel(varargin) == 5 && isstruct(varargin{1})
        data = varargin{1};
        shape = varargin{2};
        ranks = varargin{3};
        neighbors = varargin{4};
    elseif numel(varargin) == 8
        data = struct('train', varargin{1}, 'trLbl', varargin{2}, 'test', varargin{3}, 'tsLbl', varargin{4});
        shape = varargin{5};
        ranks = varargin{6};
        neighbors = varargin{7};
    else
        error('mbttda:InvalidConfig', 'Expected (data,I,ranks,K,patches) or (train,trLbl,test,tsLbl,I,ranks,K,patches).');
    end
    % Also accept already-folded patch tensors; labels still refer to observations.
    data.train = reshape(data.train, prod(shape), []);
    data.test = reshape(data.test, prod(shape), []);
    if isstruct(neighbors)
        neighbors = neighbors.K;
    end
end
