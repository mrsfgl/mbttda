function [images, labels, p, raw] = load_mda_data(dataset, dataDirectory)
    % LOAD_MDA_DATA Original MDA feature layout without full-data rank estimation.
    if nargin < 2
        dataDirectory = [];
    end
    p = mda_preset(dataset);
    raw = read_dataset_array(dataset, dataDirectory);
    if strcmp(dataset, 'YaleB')
        raw = permute(raw, [1, 2, 5, 4, 3]);
    end
    sizes = size(raw);
    p.nSamp = sizes(end - 1);
    p.class = sizes(end);
    raw = reshape(raw, [p.tensor_shape, p.nSamp, p.class]);
    images = reshape(raw, prod(p.tensor_shape), []);
    labels = repelem(1:p.class, p.nSamp);
    if ismember(char(dataset), {'GAIT', 'YaleB', 'Weizmann', 'Cambridge', 'KTH'})
        images = double(images) / 255;
    end
end
