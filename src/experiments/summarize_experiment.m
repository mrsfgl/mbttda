function summary = summarize_experiment(experiment)
    % SUMMARIZE_EXPERIMENT Comparable scalars without conflating storage definitions.
    rows = struct([]);
    for i = 1:numel(experiment.records)
        record = experiment.records(i);
        result = record.result;
        storage = NaN;
        if isfield(result, 'metrics') && isfield(result.metrics, 'storage_ratio')
            storage = result.metrics.storage_ratio;
        end
        legacy = NaN;
        if isfield(result, 'Storage')
            legacy = result.Storage(end);
        end
        seconds = NaN;
        if isfield(result, 'time_subspace')
            seconds = sum(result.time_subspace);
        end
        legacyCondition = "";
        if isfield(record.parameters, 'legacy_condition')
            legacyCondition = string(record.parameters.legacy_condition);
        end
        row = struct('dataset', string(record.dataset), 'method', string(record.method), ...
                     'repeat', record.repeat, 'train_per_class', record.train_per_class, 'ablation', record.ablation, ...
                     'tau', record.tau, 'setting', record.setting, 'rank_index', record.rank_index, ...
                     'accuracy', 1 - result.PreErr(end), 'storage_ratio', storage, 'legacy_storage', legacy, ...
                     'subspace_seconds', seconds, 'legacy_condition', legacyCondition);
        rows(end + 1) = row; %#ok<AGROW>
    end
    if isempty(rows)
        summary = table();
    else
        summary = struct2table(rows);
    end
end
