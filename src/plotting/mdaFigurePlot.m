function figures = mdaFigurePlot(~, ~, dataset, varargin)
    % MDAFIGUREPLOT Historical signature; shared plotting supports partial runs.
    figures = plot_experiment(['mdaResult_', char(dataset)]);
end
