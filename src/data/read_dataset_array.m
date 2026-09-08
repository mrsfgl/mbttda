function Data = read_dataset_array(dataset, dataDirectory)
    % READ_DATASET_ARRAY Read the Data variable; never load variables into the caller.
    if nargin < 2 || isempty(dataDirectory)
        root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
        dataDirectory = fullfile(root, 'data');
    end
    switch char(dataset)
        case 'COIL'
            filename = 'CoilData.mat';
        case 'COIL3D'
            filename = 'COIL3Ddata.mat';
        case 'MNIST'
            filename = 'MNIST.mat';
        case 'Weizmann'
            filename = 'WeizmannData.mat';
        case 'Cambridge'
            filename = 'handGestures.mat';
        case 'UCF101'
            filename = 'UCF101Data.mat';
        case 'YaleB'
            filename = 'YaleBData.mat';
        case 'GAIT'
            filename = 'GaitData.mat';
        case 'KTH'
            filename = 'KTHData.mat';
        otherwise
            error('mbttda:UnknownDataset', 'Unknown dataset: %s', dataset);
    end
    filename = fullfile(dataDirectory, filename);
    if ~isfile(filename)
        error('mbttda:MissingData', 'Expected %s. See docs/experiments.md for the MAT-file format.', filename);
    end
    loaded = load(filename, 'Data');
    if ~isfield(loaded, 'Data') || ~isnumeric(loaded.Data) || isempty(loaded.Data)
        error('mbttda:InvalidData', '%s must contain a nonempty numeric array named Data.', filename);
    end
    Data = loaded.Data;
end
