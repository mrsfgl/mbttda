function experiment = run_experiment(overrides)
    % RUN_EXPERIMENT Run explicit, reproducible method/threshold/rank/ablation sweeps.
    % config = experiment_config('COIL'); config.repeats = 1;
    % experiment = run_experiment(config);
    if nargin < 1 || ~isstruct(overrides)
        error('mbttda:InvalidConfig', 'Supply an experiment_config struct, or run demo_synthetic.');
    end
    environment = setup_mbttda;
    config = experiment_config();
    config.save_results = true;
    names = fieldnames(overrides);
    for i = 1:numel(names)
        if ~isfield(config, names{i})
            error('mbttda:InvalidConfig', 'Unknown configuration field: %s', names{i});
        end
        config.(names{i}) = overrides.(names{i});
    end
    config.methods = cellstr(upper(string(config.methods)));
    config.methods(strcmp(config.methods, '2WTTDA')) = {'TWTTDA'};
    config.methods(strcmp(config.methods, '3WTTDA')) = {'THW'};
    config.methods = unique(config.methods, 'stable');
    if isempty(config.methods)
        error('mbttda:InvalidConfig', 'Select at least one method.');
    end
    check_dependencies(config.methods);
    validateattributes(config.repeats, {'numeric'}, {'scalar', 'positive', 'integer'});
    validateattributes(config.seed, {'numeric'}, {'scalar', 'nonnegative', 'integer', 'finite'});
    if ~isempty(config.lambda)
        validateattributes(config.lambda, {'numeric'}, {'vector', 'real', 'finite'});
        if ~ismember(numel(config.lambda), [1, 3])
            error('mbttda:InvalidConfig', 'lambda must be scalar or a three-element vector.');
        end
    end
    if ~ismember(config.preset_family, {'tt', 'mda'})
        error('mbttda:InvalidConfig', 'preset_family must be tt or mda.');
    end
    previousRandomState = rng;
    restoreRandomState = onCleanup(@() rng(previousRandomState)); %#ok<NASGU>
    rng(config.seed, 'twister');

    if strcmpi(config.dataset, 'synthetic')
        [images, labels, shape] = synthetic_dataset();
        preset = struct('tensor_shape', shape, 'classsize_list', 6, 'nAbl', zeros(size(shape)));
    elseif strcmpi(config.dataset, 'custom')
        images = config.images;
        labels = config.labels;
        if isempty(images) || isempty(labels) || isempty(config.tensor_shape) || isempty(config.train_per_class)
            error('mbttda:InvalidConfig', 'Custom data requires images, labels, tensor_shape, and train_per_class.');
        end
        preset = struct('tensor_shape', config.tensor_shape, 'classsize_list', config.train_per_class, ...
                        'nAbl', zeros(size(config.tensor_shape)));
    elseif strcmp(config.preset_family, 'mda')
        [images, labels, preset] = load_mda_data(config.dataset, config.data_dir);
    else
        [images, labels, preset] = load_data(config.dataset, config.data_dir);
    end
    if ~isempty(config.tensor_shape)
        preset.tensor_shape = config.tensor_shape;
    end
    if ~isempty(config.train_per_class)
        preset.classsize_list = config.train_per_class;
    end
    if ~isempty(config.ablations)
        preset.nAbl = config.ablations;
    end
    preset.noise = config.noise;
    preset.scale = config.scale;
    shape = preset.tensor_shape;
    validateattributes(shape, {'numeric'}, {'row', 'positive', 'integer'});
    validateattributes(preset.classsize_list, {'numeric'}, {'row', 'integer', '>=', 2, 'nonempty'});
    if size(images, 1) ~= prod(shape)
        error('mbttda:InvalidShape', 'Feature count must equal prod(tensor_shape).');
    end
    if size(preset.nAbl, 2) ~= numel(shape)
        error('mbttda:InvalidConfig', 'Each ablation row needs one entry per tensor mode.');
    end
    validateattributes(preset.nAbl, {'numeric'}, {'2d', 'integer', 'nonnegative'});
    if ~isempty(config.ranks)
        validateattributes(config.ranks, {'numeric'}, {'2d', 'positive', 'integer', 'finite'});
        rankCheck = config.ranks;
        if isvector(rankCheck)
            rankCheck = rankCheck(:);
        end
        if size(rankCheck, 1) ~= numel(shape) || any(any(rankCheck > shape(:)))
            error('mbttda:InvalidRank', 'Supply one valid rank per physical mode; rank-sweep settings are columns.');
        end
    end
    counts = arrayfun(@(label) sum(labels == label), unique(labels));
    if any(preset.classsize_list >= min(counts))
        error('mbttda:InsufficientSamples', 'Every training size must leave test samples in every class.');
    end
    for method = config.methods
        method_tau_grid(method{1}, config, preset); % Fail before starting a partially specified sweep.
    end

    if config.save_results
        if isempty(config.output_dir)
            config.output_dir = fullfile(environment.root, 'results', datestr(now, 'yyyymmdd_HHMMSS_FFF'));
        end
        if isfolder(config.output_dir) && ~isempty(dir(fullfile(config.output_dir, '*.mat')))
            error('mbttda:ExistingResults', 'Choose an output directory without existing MAT files.');
        end
        if ~isfolder(config.output_dir)
            mkdir(config.output_dir);
        end
    end
    experiment = struct('format_version', 1, 'config', rmfield(config, {'images', 'labels'}), ...
                        'environment', environment, 'records', struct([]));
    experiment.environment.run_revision = git_revision(environment.root);
    experiment.metric_notes = struct('storage', 'Element ratio and legacy storage are distinct', ...
                                     'subspace_time', 'Elapsed seconds including initialization; excludes validation, embedding, and classification', ...
                                     'mda_ranks', 'Automatic ranks use training data only');
    metadata = rmfield(experiment, 'records');
    for repeat = 1:config.repeats
        for sizeIndex = 1:numel(preset.classsize_list)
            data = get_data(images, labels, preset, sizeIndex);
            data = validate_dataset(data, shape);
            trainingPerClass = preset.classsize_list(sizeIndex);
            lambda = config.lambda;
            validation = struct();
            neededBranches = [];
            for method = config.methods
                branch = lambda_branch(method{1});
                if branch > 0
                    neededBranches(end + 1) = branch;
                end %#ok<AGROW>
            end
            if isempty(lambda) && ~isempty(neededBranches)
                options = config.validation;
                options.methods = unique(neededBranches);
                [lambda, ~, data, validation] = optLambda(data, shape, trainingPerClass, config.validation_per_class, options);
            elseif isempty(lambda)
                lambda = ones(1, 3);
            end
            if isscalar(lambda)
                lambda = repmat(lambda, 1, 3);
            end
            if numel(lambda) ~= 3
                error('mbttda:InvalidConfig', 'lambda must be scalar or a three-element vector.');
            end
            split = struct('train', data.train_indices, 'test', data.test_indices);
            for ablation = 1:size(preset.nAbl, 1)
                for methodIndex = 1:numel(config.methods)
                    method = config.methods{methodIndex};
                    taus = method_tau_grid(method, config, preset);
                    rankGrid = config.ranks;
                    if isempty(rankGrid)
                        rankGrid = zeros(0, 1);
                    elseif isvector(rankGrid)
                        rankGrid = rankGrid(:);
                    end
                    if ~ismember(method, {'CMDA', 'DGTDA'})
                        rankGrid = zeros(0, 1);
                    end
                    if ~isempty(rankGrid)
                        taus = 0;
                    end % Explicit ranks replace threshold-based rank selection.
                    for tauIndex = 1:numel(taus)
                        for rankIndex = 1:size(rankGrid, 2)
                            p = algorithm_parameters(shape, config.parameters);
                            p.I = shape;
                            p.nA = preset.nAbl(ablation, :);
                            p.tau = taus(tauIndex);
                            branch = lambda_branch(method);
                            if branch > 0
                                p.lambda = lambda(branch);
                            end
                            p = algorithm_parameters(shape, p);
                            ranks = rankGrid(:, rankIndex)';
                            result = run_method(method, data, p, ranks);
                            if isfield(result, 'ranks')
                                ranks = result.ranks;
                            end
                            record = struct('dataset', config.dataset, 'method', method, 'repeat', repeat, ...
                                            'train_per_class', trainingPerClass, 'ablation', ablation, 'tau', p.tau, ...
                                            'setting', tauIndex, 'rank_index', rankIndex, 'ranks', ranks, 'parameters', p, ...
                                            'split', split, 'validation', validation, 'result', result);
                            experiment.records(end + 1) = record; %#ok<AGROW>
                            if config.save_results
                                filename = sprintf('run_%03d_size_%03d_abl_%03d_%s_tau_%03d_rank_%03d.mat', ...
                                                   repeat, sizeIndex, ablation, method, tauIndex, rankIndex);
                                save(fullfile(config.output_dir, filename), 'record', 'metadata', '-v7');
                            end
                        end
                    end
                end
            end
        end
    end
    experiment.summary = summarize_experiment(experiment);
    if config.save_results
        save(fullfile(config.output_dir, 'experiment.mat'), 'experiment', '-v7.3');
    end
    if config.plot
        plot_experiment(experiment, config.figure_visible);
    end
end

function branch = lambda_branch(method)
    switch method
        case {'TTDA', 'BTT', 'LDA'}
            branch = 1;
        case {'TWTTDA', 'TWBTT', 'THWBTT'}
            branch = 2; % Preserve mydemo's ThWBTT preset.
        case 'THW'
            branch = 3;
        otherwise
            branch = 0;
    end
end

function revision = git_revision(root)
    % Read ordinary clone metadata without invoking a shell or changing pwd.
    revision = 'unavailable';
    head = fullfile(root, '.git', 'HEAD');
    if ~isfile(head)
        return
    end
    value = strtrim(fileread(head));
    if startsWith(value, 'ref: ')
        ref = extractAfter(value, 'ref: ');
        loose = fullfile(root, '.git', char(ref));
        if isfile(loose)
            revision = strtrim(fileread(loose));
        elseif isfile(fullfile(root, '.git', 'packed-refs'))
            lines = splitlines(string(fileread(fullfile(root, '.git', 'packed-refs'))));
            match = lines(endsWith(lines, " " + string(ref)));
            if ~isempty(match)
                revision = char(extractBefore(match(1), " "));
            end
        end
    else
        revision = value;
    end
end
