function [labels, errors] = Classfier_KNN(train, trainLabels, test, testLabels, neighbors)
    % CLASSFIER_KNN Historical spelling retained for compatibility.
    % Samples are columns; returned predictions are test-samples-by-K-settings.
    trainLabels = trainLabels(:);
    testLabels = testLabels(:);
    validateattributes(neighbors, {'numeric'}, {'vector', 'integer', 'positive', '<=', size(train, 2)});
    if numel(trainLabels) ~= size(train, 2) || numel(testLabels) ~= size(test, 2)
        error('mbttda:InvalidLabels', 'Labels must match sample counts.');
    end
    labels = zeros(size(test, 2), numel(neighbors));
    errors = zeros(numel(neighbors), 1);
    for i = 1:numel(neighbors)
        classifier = fitcknn(train', trainLabels, 'NumNeighbors', neighbors(i));
        labels(:, i) = predict(classifier, test');
        errors(i) = mean(labels(:, i) ~= testLabels);
    end
end
