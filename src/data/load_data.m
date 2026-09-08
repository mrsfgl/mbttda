function [images, labels, p] = load_data(dataset, dataDirectory)
    % LOAD_DATA Original TT dataset layout and experiment preset.
    % images: features-by-samples; labels: one numeric label per sample column.
    if nargin < 2
        dataDirectory = [];
    end
    p = dataset_preset(dataset);
    Data = read_dataset_array(dataset, dataDirectory);
    switch char(dataset)
        case 'YaleB'
            Data = permute(Data, [1, 2, 5, 4, 3]);
        case 'COIL'
            % Preserve the original bit-interleaving of the two image axes.
            order = reshape(reshape(1:12, 6, 2)', 1, 12);
            Data = permute(reshape(Data, [2 * ones(1, 12), 72, 100]), [order, 13, 14]);
    end
    shape = size(Data);
    p.class = shape(end);
    p.nSamp = shape(end - 1);
    images = reshape(Data, [], p.class * p.nSamp);
    labels = repelem(1:p.class, p.nSamp);
    if size(images, 1) ~= prod(p.tensor_shape)
        error('mbttda:InvalidShape', 'Data feature count does not match the %s preset.', dataset);
    end
    % Historical loader normalization is distinct from get_data's scaling.
    if ismember(char(dataset), {'Cambridge', 'KTH'})
        images = double(images) / 255;
    end
    if strcmp(dataset, 'Weizmann')
        images = double(images);
    end
end
