function [within, between] = const_scat(data, varargin)
    % CONST_SCAT Original covariance-normalized scatter convention.
    % Layout: [features, samples per class, classes, projected coordinates].
    % Extra legacy size arguments are accepted for the BTT callers.
    % Unlike unnormalized scatter sums, cov divides by samples-1 (and classes-1).
    features = size(data, 1);
    samples = size(data, 2);
    classes = size(data, 3);
    if samples < 2 || classes < 2
        error('mbttda:InvalidScatter', 'Scatter requires at least two samples per class and two classes.');
    end
    projected = numel(data) / (features * samples * classes);
    data = reshape(data, features, samples, classes, projected);
    within = zeros(features);
    between = zeros(features);
    for coordinate = 1:projected
        means = zeros(features, classes);
        for classId = 1:classes
            observations = data(:, :, classId, coordinate);
            means(:, classId) = mean(observations, 2);
            centered = observations - means(:, classId);
            within = within + (centered * centered') / (samples - 1);
        end
        centeredMeans = means - mean(means, 2);
        between = between + (centeredMeans * centeredMeans') / (classes - 1);
    end
end
