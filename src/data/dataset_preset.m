function p = dataset_preset(dataset)
    % DATASET_PRESET Original TT experiment values, separated from data I/O.
    % Values are from revision 189e39f. Missing options stay explicit.
    p = struct();
    switch char(dataset)
        case 'YaleB'
            p.K_list          = 1;
            p.tau_list        = [0.21:-0.03:0.06, 0.05:-0.01:0.03];
            p.tau_list2       = [0.21:-0.03:0.06, 0.05:-0.015:0.02] * 2;
            p.tau_list3       = [0.21:-0.03:0.06, 0.05:-0.015:0.02] * 2;  % threshold parameter for 3WTT
            p.lambda_list     = [1e3];
            p.lambda_list2     = 10.^[0];
            p.tensor_shape    = [30, 40, 20]; % the dimension for dataset dependent reshaped tensors
            p.classsize_list  = [32];
        case 'GAIT'
            p.K_list          = 1;
            p.tau_list        = [0.21:-0.03:0.06, 0.05:-0.01:0.03];
            p.tau_list2       = [0.21:-0.03:0.06, 0.05:-0.015:0.02] * 2;
            p.tau_list3       = [0.21:-0.03:0.06, 0.05:-0.015:0.02] * 2;  % threshold parameter for 3WTT
            p.lambda_list     = [1];
            p.lambda_list2    = [1];
            p.tensor_shape    = [10, 6, 11, 8, 10, 5]; % the dimension for dataset dependent reshaped tensors
            p.classsize_list  = 38;
        case 'COIL'
            p.K_list = 1;
            p.tau_list = [0.6, 0.4:-0.09:0.04, 0.03, 0.02, 0.01, 0];
            p.tau_list2 = [0.3:-0.05:0.15, 0.12, 0.1, 0.09, 0.07, 0.04, 0];
            p.tau_list3 = [0.3, 0.25, 0.2, 0.15, 0.12, 0.1, 0.08, 0.04, 0.02, 0];
            p.tau_list4 = [0.3:-0.05:0.15, 0.12, 0.1, 0.09, 0.06,  0.03, 0];
            p.lambda_list = 1;
            p.lambda_list2 = 1;
            p.tensor_shape = [4, 4, 4, 4, 4, 4]; % the dimension for dataset dependent reshaped tensors
            p.classsize_list = [20];
            p.nAbl = [0, 0, 0, 0, 0, 0];
        case 'MNIST'
            p.K_list          = 1;
            p.tau_list        = [0.7, 0.6:-0.08:0.12, 0.03, 0.01, 0];    % threshold parameter for TTNPE
            p.tau_list2        = [0.48:-0.02:0.28];                 % threshold parameter for 2WTT
            p.tau_list3        = [0.9:-0.1:0.5, 0.4:-0.09:0.04, 0.03];   % threshold parameter for 3WTT
            p.lambda_list     = 1;
            p.lambda_list2    = 1;
            p.tensor_shape    = [4, 7, 4, 7]; % the dimension for dataset dependent reshaped tensors
            p.classsize_list  = [100];
        case 'COIL3D'
            p.K_list          = 1;
            p.tau_list        = [0.5:-0.05:0.15, 0]; % threshold parameter for TTNPE
            p.tau_list2        = [0.3:-0.04:0.02, 0]; % threshold parameter for TTLDA
            p.tau_list3        = [0.3:-0.04:0.02, 0]; % threshold parameter for TTLDA
            p.tau_list4        = [0.3:-0.04:0.02, 0];
            p.lambda_list     = [1, 1, 1];
            p.lambda_list2     = 1;
            p.tensor_shape    = [64, 64, 8]; % the dimension for dataset dependent reshaped tensors
            p.classsize_list  = [3];
            p.nAbl            = [0, 0, 0];
        case 'Weizmann'
            p.K_list          = 1;
            p.tau_list        = [0.16, 0.156, .155, .152, 0.15:-0.03:0.06, (0.05:-0.02:0.01) / 2] * 2;
            p.tau_list2       = [0.16, 0.156, .155, .152, 0.15:-0.03:0.06, (0.05:-0.02:0.01) / 2] * 2;
            p.tau_list3       = [0.16, 0.156, .155, .152, 0.15:-0.03:0.06, 0.05:-0.015:0.02];
            p.tau_list4       = [0.16, 0.156, .155, .152, 0.15:-0.03:0.06, 0.05:-0.015:0.02];  % threshold parameter for 3WTT
            p.lambda_list     = [1e3];
            p.lambda_list2    = 10.^[0];
            p.tensor_shape    = [8, 8, 44]; % the dimension for dataset dependent reshaped tensors
            p.classsize_list  = [20];
            p.nAbl            = [0, 0, 0]; % Repair zero ablation width to match the 3-mode preset.
        case 'UCF101'
            p.classsize_list  = [10:10:90];
            p.K_list          = 1;
            p.tau_list        = [0.01];
            p.tau_list3        = [0.01];
            p.tau_list4       = [0.12];   % threshold parameter for 3WTT
            p.tensor_shape    = [5, 6, 5, 8, 5, 10]; % the dimension for dataset dependent reshaped tensors
            p.nAbl            = [0, 0, 0, 0, 0, 0];
        case 'Cambridge'
            p.classsize_list  = [4];
            p.K_list          = 1;
            p.tau_list        = [0.28:-0.04:0.20, 0.12, 0.08, 0.06, 0.04, 0.03:-0.01:0.01];
            p.tau_list2       = [0.28:-0.04:0.20, 0.12, 0.08, 0.06, 0.04, 0.03:-0.01:0.01];
            p.tau_list3       = [0.28, 0.24, 0.16:-0.04:0.04, 0.03:-0.01:0];   % threshold parameter for 3WTT
            p.tensor_shape    = [30, 40, 30, 10]; % the dimension for dataset dependent reshaped tensors
            p.nAbl            = [0, 0, 0, 0];
        case 'KTH'
            p.classsize_list  = [190];
            p.K_list          = 1;
            p.tau_list        = [0.4:-0.04:0.04, 0.03:-0.01:0.01];
            p.tau_list2       = [0.3:-0.03:0.06, 0.05:-0.01:0.02] / 3;
            p.tau_list3       = [0.4:-0.04:0.04, 0.03:-0.01:0.01] / 4;   % threshold parameter for 3WTT
            p.lambda_list     = 1;
            p.lambda_list2    = 10.^[0];
            p.tensor_shape    = [30, 40, 30]; % the dimension for dataset dependent reshaped tensors
        otherwise
            error('mbttda:UnknownDataset', 'Unknown dataset: %s', dataset);
    end
    % Only fill absent plumbing, never replace existing numerical values.
    if ~isfield(p, 'nAbl')
        p.nAbl = zeros(1, numel(p.tensor_shape));
    end
    % These incomplete presets require an explicit tau override for that method.
    if ~isfield(p, 'tau_list2')
        p.tau_list2 = [];
    end
    if ~isfield(p, 'tau_list3')
        p.tau_list3 = [];
    end
    if ~isfield(p, 'tau_list4')
        p.tau_list4 = [];
    end
    p.noise = 0;
    p.repeats = 10;
end
