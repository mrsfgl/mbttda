function output = wwq_squeeze(input)
    % WWQ_SQUEEZE Remove singleton axes, using a column for a scalar/vector result.
    shape = size(input);
    shape = shape(shape ~= 1);
    if numel(shape) < 2
        output = input(:);
    else
        output = reshape(input, shape);
    end
end
