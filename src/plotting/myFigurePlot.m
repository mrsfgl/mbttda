function figures = myFigurePlot(~, ~, ~, ~, dataset, varargin)
    % MYFIGUREPLOT Historical signature, replacing the incomplete plotting script.
    figures = plot_experiment(['result', char(dataset)]);
end
