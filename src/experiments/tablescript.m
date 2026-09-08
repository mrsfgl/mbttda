function summary = tablescript(source)
    % TABLESCRIPT Return a table from a results folder or a saved experiment.
    summary = summarize_experiment(load_experiment(source));
    disp(summary);
end
