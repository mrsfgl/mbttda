function [images, labels, rankList, tensorShape, imageShape, classCount, classSizes] = load_dataMDA(dataset, withRanks, dataDirectory)
    % LOAD_DATAMDA Historical seven-output API, including original full-data ranks.
    % Modern run_experiment estimates automatic MDA ranks on training data only.
    if nargin < 2
        withRanks = false;
    end
    if nargin < 3
        dataDirectory = [];
    end
    [images, labels, p, raw] = load_mda_data(dataset, dataDirectory);
    tensorShape = p.tensor_shape;
    imageShape = tensorShape;
    classCount = p.class;
    classSizes = p.classsize_list;
    if ~withRanks
        rankList = zeros(numel(tensorShape), numel(p.tau_list));
        for i = 1:numel(p.tau_list)
            ranks = estimate_mda_ranks(reshape(double(raw), prod(tensorShape), []), tensorShape, p.tau_list(i));
            rankList(:, i) = ranks(:);
        end
    else
        switch char(dataset)
            case {'COIL', 'COIL3D', 'MNIST'}
                rankList = (tensorShape(:) / 8) * (2:8);
            case {'Weizmann', 'GAIT', 'YaleB'}
                rankList = [1, 2, 4, 8, 16, 32, 64; 1, 2, 4, 6, 10, 16, 44; 1, 2, 2, 3, 3, 4, 5];
            case {'Cambridge', 'KTH'}
                rankList = [2, 3, 6, 15, 20; 2, 4, 8, 20, 30; 2, 3, 6, 15, 20; 2, 3, 4, 5, 6];
        end
        if size(rankList, 1) ~= numel(tensorShape) || any(rankList(:) ~= floor(rankList(:))) || any(any(rankList > tensorShape(:)))
            error('mbttda:InvalidLegacyPreset', 'The original fixed-rank table is incompatible with this tensor shape. Use withRanks=false or explicit valid ranks in run_experiment.');
        end
    end
end
