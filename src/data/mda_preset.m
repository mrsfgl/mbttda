function p = mda_preset(dataset)
    % MDA_PRESET Numerical Tucker presets separated from data and rank estimation.
    p = struct('noise', 0, 'repeats', 10, 'K_list', 1);
    switch char(dataset)
        case 'COIL'
            p.tensor_shape = [8, 8, 8, 8];
            p.classsize_list = 20;
            p.tau_list = [0.95, 0.9:-0.2:0.5, 0.3, 0.12, 0.04, 0.03:-0.02:0.01, 0];
        case 'COIL3D'
            p.tensor_shape = [64, 64, 8];
            p.classsize_list = 3;
            p.tau_list = [0.3, 0.2:-0.04:0.04, 0.03:-0.02:0.01, 0];
        case 'MNIST'
            p.tensor_shape = [4, 7, 4, 7];
            p.classsize_list = 100;
            p.tau_list = [0.6:-0.1:0.5, 0.4, 0.3, 0.2:-0.08:0.04, 0.03:-0.01:0.01, 0];
        case 'Weizmann'
            p.tensor_shape = [4, 4, 4, 4, 11];
            p.classsize_list = 20;
            p.tau_list = [0.24, 0.18:-0.03:0.06, 0.05:-0.015:0.005];
        case 'Cambridge'
            p.tensor_shape = [30, 40, 30, 10];
            p.classsize_list = 4;
            p.tau_list = [0.3, 0.2:-0.08:0.04, 0.03:-0.01:0.01, 0];
        case 'YaleB'
            p.tensor_shape = [30, 40, 20];
            p.classsize_list = 32;
            p.tau_list = [0.24:-0.03:0.06, 0.05:-0.015:0.02] * 2;
        case 'GAIT'
            p.tensor_shape = [10, 6, 11, 8, 10, 5];
            p.classsize_list = 38;
            p.tau_list = [0.24:-0.03:0.06, 0.05:-0.015:0.02] * 2;
        case 'KTH'
            p.tensor_shape = [30, 40, 30];
            p.classsize_list = 190;
            p.tau_list = [0.09, 0.06, 0.04:-0.005:0.01, 0];
        otherwise
            error('mbttda:MissingPreset', 'No historical MDA preset for %s. Use the TT layout or supply custom data.', dataset);
    end
    p.nAbl = zeros(size(p.tensor_shape));
end
