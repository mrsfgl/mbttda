function varargout = get_data(images, labels, varargin)
    % GET_DATA Balanced random split with original image scaling and optional noise.
    % data = get_data(images,labels,preset,sizeIndex)
    % [train,trLbl,test,tsLbl] = get_data(images,labels,C,K,noise)
    if numel(varargin) == 2 && isstruct(varargin{1})
        p = varargin{1};
        perClass = p.classsize_list(varargin{2});
        noise = p.noise;
        scale = 255;
        if isfield(p, 'scale')
            scale = p.scale;
        end
    elseif numel(varargin) == 3
        perClass = varargin{2};
        noise = varargin{3};
        scale = 255;
    else
        error('mbttda:InvalidConfig', 'Use a preset and index, or class count, training count, and noise.');
    end
    validateattributes(perClass, {'numeric'}, {'scalar', 'integer', '>=', 2});
    validateattributes(scale, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
    validateattributes(noise, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'});
    labels = reshape(labels, 1, []);
    if numel(labels) ~= size(images, 2)
        error('mbttda:InvalidLabels', 'Each sample column must have one label.');
    end
    classes = unique(labels);
    trainIndex = [];
    testIndex = [];
    for c = classes
        available = find(labels == c);
        if numel(available) <= perClass
            error('mbttda:InsufficientSamples', 'Class %g needs more than %d samples.', c, perClass);
        end
        chosen = available(randperm(numel(available), perClass));
        trainIndex = [trainIndex, chosen]; %#ok<AGROW>
        testIndex = [testIndex, setdiff(available, chosen, 'stable')]; %#ok<AGROW>
    end
    data.train = double(images(:, trainIndex)) / scale;
    data.test = double(images(:, testIndex)) / scale;
    data.trLbl = labels(trainIndex);
    data.tsLbl = labels(testIndex);
    data.train_indices = trainIndex;
    data.test_indices = testIndex;
    if noise > 0
        trainNoise = randn(size(data.train));
        testNoise = randn(size(data.test));
        data.train = data.train + noise * trainNoise / max(norm(trainNoise(:)), eps);
        data.test = data.test + noise * testNoise / max(norm(testNoise(:)), eps);
    end
    if nargout <= 1
        varargout = {data};
    else
        varargout = {data.train, data.trLbl, data.test, data.tsLbl};
    end
end
