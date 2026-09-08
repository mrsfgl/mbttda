function result = main_App(varargin)
    % MAIN_APP TTNPE classification, supporting the demo and historical signatures.
    setup_mbttda;
    if nargin == 2 && isstruct(varargin{1})
        data = varargin{1};
        p = algorithm_parameters(varargin{2}.I, varargin{2});
    elseif nargin == 9
        data = struct('train', varargin{1}, 'trLbl', varargin{2}, 'test', varargin{3}, 'tsLbl', varargin{4});
        p = algorithm_parameters(varargin{5}, varargin{8});
        p.tau = varargin{6};
        p.Graph = varargin{7};
        neighbors = varargin{9};
        if isstruct(neighbors)
            neighbors = neighbors.K;
        end
        if ~isequal(neighbors, 1)
            error('mbttda:InvalidConfig', 'The TTNPE compatibility entry point supports 1-NN.');
        end
    else
        error('mbttda:InvalidConfig', 'Use main_App(data,parameters), or the historical nine arguments.');
    end
    result = ttnpe_classifier(data, algorithm_parameters(p.I, p));
end
