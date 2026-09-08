function vectors = smallest_eigenvectors(matrix, count)
    % SMALLEST_EIGENVECTORS Ordered minimizers of a symmetric trace objective.
    matrix = (matrix + matrix') / 2;
    validateattributes(count, {'numeric'}, {'scalar', 'integer', 'positive', '<=', size(matrix, 1)});
    if count >= size(matrix, 1) - 1
        [vectors, diagonal] = eig(matrix);
    else
        [vectors, diagonal] = eigs(matrix, count, 'smallestreal');
    end
    [~, order] = sort(diag(diagonal), 'ascend');
    vectors = vectors(:, order(1:count));
end
