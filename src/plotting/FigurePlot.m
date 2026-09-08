function figures = FigurePlot(~, ~, ~, ~, dataset, varargin)
    % FIGUREPLOT Historical signature; reads available Result_<dataset> files.
    figures = plot_experiment(['Result_', char(dataset)]);
end
