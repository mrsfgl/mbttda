function experiment = load_experiment(source)
    % LOAD_EXPERIMENT Read a modern run, partial checkpoints, or historical MAT files.
    if isstruct(source)
        experiment = source;
        return
    end
    source = char(source);
    if isfolder(source)
        complete = fullfile(source, 'experiment.mat');
        if isfile(complete)
            experiment = load_experiment(complete);
            return
        end
        files = dir(fullfile(source, '*.mat'));
        filenames = arrayfun(@(f) fullfile(f.folder, f.name), files, 'UniformOutput', false);
    else
        filenames = {source};
    end
    experiment = struct('format_version', 1, 'records', struct([]));
    if isempty(filenames)
        error('mbttda:MissingResults', 'No MAT results found in %s.', source);
    end
    for i = 1:numel(filenames)
        loaded = load(filenames{i});
        if isfield(loaded, 'experiment')
            if numel(filenames) == 1
                experiment = loaded.experiment;
                experiment.summary = summarize_experiment(experiment);
                return
            end
            records = loaded.experiment.records;
            metadata = rmfield(loaded.experiment, intersect(fieldnames(loaded.experiment), {'records', 'summary'}));
        elseif isfield(loaded, 'record')
            records = loaded.record;
            if isfield(loaded, 'metadata')
                metadata = loaded.metadata;
            else
                metadata = struct();
            end
        else
            records = legacy_records(loaded, filenames{i});
            metadata = struct();
        end
        fields = fieldnames(metadata);
        for field = 1:numel(fields)
            name = fields{field};
            if isfield(experiment, name) && ~isequaln(experiment.(name), metadata.(name))
                error('mbttda:IncompatibleResults', 'Result files contain different %s metadata. Load each run separately.', name);
            end
            experiment.(name) = metadata.(name);
        end
        if ~isempty(records)
            experiment.records = [experiment.records, records];
        end %#ok<AGROW>
    end
    if isempty(experiment.records)
        error('mbttda:MissingResults', 'No classification results found.');
    end
    experiment.summary = summarize_experiment(experiment);
end

function records = legacy_records(loaded, filename)
    records = struct([]);
    names = fieldnames(loaded);
    [folder, basename] = fileparts(filename);
    [~, dataset] = fileparts(folder);
    if isempty(dataset)
        dataset = 'legacy';
    end
    % Old files encode a ratio/index, not a reliable training sample count.
    % Keep the exact holdout/shape text so different settings cannot be averaged.
    condition = regexp(basename, '_(?:holdout|hout)[^_]+', 'match', 'once');
    shape = regexp(basename, '_shape.+$', 'match', 'once');
    parameters = struct('source_file', filename, 'legacy_condition', [condition, shape]);
    for i = 1:numel(names)
        result = loaded.(names{i});
        if ~isstruct(result)
            continue
        end
        if ~isfield(result, 'PreErr') && isfield(result, 'LDAPreErr')
            result.PreErr = result.LDAPreErr;
            if isfield(result, 'LDAStorage')
                result.Storage = result.LDAStorage;
            end
            if isfield(result, 'LDAtime_subspace')
                result.time_subspace = result.LDAtime_subspace;
            end
        end
        if ~isfield(result, 'PreErr')
            continue
        end
        record = struct('dataset', dataset, 'method', names{i}, ...
                        'repeat', filename_number(filename, 'repeat', 1), 'train_per_class', NaN, ...
                        'ablation', filename_number(filename, 'ablation', 1), ...
                        'tau', filename_number(filename, 'tau', NaN), ...
                        'setting', filename_number(filename, 'tau', filename_number(filename, 'ranks', 0)), ...
                        'rank_index', 1, 'ranks', [], 'parameters', parameters, ...
                        'split', struct(), 'validation', struct(), 'result', result);
        records(end + 1) = record; %#ok<AGROW>
    end
end

function value = filename_number(filename, key, fallback)
    [~, basename] = fileparts(filename);
    token = regexp(basename, ['_', key, '([-+0-9.eE]+)'], 'tokens', 'once');
    if isempty(token)
        value = fallback;
    else
        value = str2double(token{1});
    end
end
