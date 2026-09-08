function [bestLambda, accuracy, data, validation] = optLambda(data, shape, trainingPerClass, validationPerClass, options)
    % OPTLAMBDA Original leave-s-out search, with recorded splits and final holdout.
    % Optional settings: candidates, repeats, maxiter, tau, methods (branch IDs).
    if nargin < 5
        options = struct();
    end
    defaults = struct('candidates', 10.^(-2:4), 'repeats', 5, 'maxiter', 50, 'tau', 0.2, 'methods', 1:3);
    names = fieldnames(options);
    for i = 1:numel(names)
        if ~isfield(defaults, names{i})
            error('mbttda:InvalidConfig', 'Unknown validation option: %s', names{i});
        end
        defaults.(names{i}) = options.(names{i});
    end
    options = defaults;
    validateattributes(options.candidates, {'numeric'}, {'vector', 'positive', 'finite'});
    validateattributes(options.repeats, {'numeric'}, {'scalar', 'integer', 'positive'});
    validateattributes(options.methods, {'numeric'}, {'vector', 'integer', '>=', 1, '<=', 3});
    data = validate_dataset(data, shape);
    counts = arrayfun(@(c) sum(data.trLbl == c), unique(data.trLbl));
    if any(counts ~= trainingPerClass)
        error('mbttda:InvalidConfig', 'trainingPerClass must match the input.');
    end
    [data, allocation] = reserve_validation(data, validationPerClass);
    methods = {@ttda, @twttda, @thwttda};
    p = algorithm_parameters(shape, struct('maxiter', options.maxiter, 'tau', options.tau));
    accuracy = nan(options.repeats, numel(options.candidates), 3);
    validation = struct('allocation', allocation, 'options', options, 'folds', {{}});
    for repeat = 1:options.repeats
        fold = validation_fold(data, trainingPerClass);
        validation.folds{repeat} = struct('train', fold.train_indices, 'validation', fold.test_indices);
        for candidate = 1:numel(options.candidates)
            p.lambda = options.candidates(candidate);
            for branch = options.methods
                result = methods{branch}(fold, p);
                accuracy(repeat, candidate, branch) = 1 - result.PreErr(end);
            end
        end
    end
    bestLambda = nan(1, 3);
    for branch = options.methods
        scores = mean(accuracy(:, :, branch), 1);
        % Original sort selected the final candidate among equal best scores.
        index = find(scores == max(scores), 1, 'last');
        bestLambda(branch) = options.candidates(index);
    end
    validation.accuracy = accuracy;
    validation.selected_lambda = bestLambda;
end
