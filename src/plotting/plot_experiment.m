function figures = plot_experiment(source, visible)
    % PLOT_EXPERIMENT Mean accuracy and training time versus storage, grouped by run.
    % Legacy byte-based storage is plotted separately from paper element ratios.
    if nargin < 2
        visible = 'on';
    end
    experiment = load_experiment(source);
    summary = summarize_experiment(experiment);
    if isempty(summary)
        error('mbttda:MissingResults', 'No results to plot.');
    end
    legacy = isnan(summary.storage_ratio);
    groups = summary.dataset + " / train=" + string(summary.train_per_class) + ...
        " / ablation=" + string(summary.ablation) + " / legacy=" + string(legacy) + summary.legacy_condition;
    keys = unique(groups, 'stable');
    figures = gobjects(numel(keys), 1);
    for group = 1:numel(keys)
        rows = summary(groups == keys(group), :);
        isLegacy = all(isnan(rows.storage_ratio));
        figures(group) = figure('Visible', visible, 'Name', char(keys(group)));
        layout = tiledlayout(figures(group), 1, 2);
        accuracyAxes = nexttile(layout);
        hold(accuracyAxes, 'on');
        timeAxes = nexttile(layout);
        hold(timeAxes, 'on');
        methods = unique(rows.method, 'stable');
        for method = methods'
            selected = rows(rows.method == method, :);
            settings = unique([selected.setting, selected.rank_index], 'rows', 'stable');
            storage = nan(size(settings, 1), 1);
            accuracy = storage;
            seconds = storage;
            for setting = 1:size(settings, 1)
                members = selected.setting == settings(setting, 1) & selected.rank_index == settings(setting, 2);
                if isLegacy
                    storage(setting) = mean(selected.legacy_storage(members), 'omitnan');
                else
                    storage(setting) = mean(selected.storage_ratio(members), 'omitnan');
                end
                accuracy(setting) = mean(selected.accuracy(members), 'omitnan');
                seconds(setting) = mean(selected.subspace_seconds(members), 'omitnan');
            end
            [storage, order] = sort(storage);
            plot(accuracyAxes, storage, accuracy(order), '-o', 'DisplayName', char(method));
            plot(timeAxes, storage, seconds(order), '-o', 'DisplayName', char(method));
        end
        if isLegacy
            label = 'Legacy storage (original units)';
        else
            label = 'Normalized storage (element ratio)';
        end
        xlabel(accuracyAxes, label);
        ylabel(accuracyAxes, 'Classification accuracy');
        ylim(accuracyAxes, [0, 1]);
        xlabel(timeAxes, label);
        ylabel(timeAxes, 'Subspace training time (s)');
        legend(accuracyAxes, 'Location', 'best');
        legend(timeAxes, 'Location', 'best');
        grid(accuracyAxes, 'on');
        grid(timeAxes, 'on');
    end
    if any(isnan(summary.subspace_seconds))
        warning('mbttda:IncompleteResults', 'Some results lack training timings; those points are omitted.');
    end
end
