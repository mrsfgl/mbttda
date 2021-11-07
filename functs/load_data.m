function [images, labels, param] = load_data(Dataset)
% (Input)   Dataset:        a string of the name of dataset
% (Output)  images:         d * N   vectorized dataset
%           labels:         1 * N   labels for each data
%           param.K_list         : A vector, each entry is the amount of
%                           neighbors to build the graph and conduct KNN
%                           classification.
%           param.tau_list*      : A vector, each entry is the
%                           thresholding parameter for TT
%           param.lamdba_list*   : A vector, each entry is the balance
%                           parameter for LDA(deprecated, is decided by
%                           validation now.)
%           param.tensor_shape   : The dimension for dataset dependent
%                           reshaped tensors.
%           param.classsize_list : A vector, where each element is the 
%                           number of samples in each class.
%           param.class          : A scalar that indicates the number of
%                           classes for each data.
%           param.nAbl           : A matrix where each row indicates how
%                           many random slices of factors at the mode 
%                           corresponding to the column index will be set 
%                           to zero for ablation.
%
% Inherited from TTNPE toolbox. Added new data and parameters.
if strcmp(Dataset,'YaleB')
    % Yale B Facebase
    % Deprecated
    load YaleBData.mat
    Data            = permute(Data,[1,2,5,4,3]);
    img_size        = size(Data);
    images          = reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]);
    labels          = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class           = img_size(ndims(Data));
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list          = 1;
    param.tau_list        = [0.21:-0.03:0.06,0.05:-0.01:0.03];
    param.tau_list2       = [0.21:-0.03:0.06,0.05:-0.015:0.02]*2;
    param.tau_list3       = [0.21:-0.03:0.06,0.05:-0.015:0.02]*2;  %threshold parameter for 3WTT
    param.lambda_list     = [1e3];
    param.lambda_list2     = 10.^[0];
    param.tensor_shape    = [30,40,20];% the dimension for dataset dependent reshaped tensors
    param.classsize_list  = [32];
elseif strcmp(Dataset, 'GAIT')
    % Deprecated
    load GaitData.mat
    img_size        = size(Data);
    images          = reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]);
    labels          = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class     = img_size(ndims(Data));
    param.K_list          = 1;
    param.tau_list        = [0.21:-0.03:0.06,0.05:-0.01:0.03];
    param.tau_list2       = [0.21:-0.03:0.06,0.05:-0.015:0.02]*2;
    param.tau_list3       = [0.21:-0.03:0.06,0.05:-0.015:0.02]*2;  %threshold parameter for 3WTT
    param.lambda_list     = [1];
    param.lambda_list2    = [1];
    param.tensor_shape    = [10,6,11,8,10,5];% the dimension for dataset dependent reshaped tensors
    param.classsize_list  = 38;
elseif strcmp(Dataset,'COIL')
    % COIL-100
    load CoilData.mat
    a = reshape(reshape(1:12, 6,2)',1,12);
    Data = permute(reshape(Data, [2*ones(1,12), 72, 100]), [a,13,14]);
    img_size = size(Data);
    images = reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]);
    labels = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class = img_size(ndims(Data));
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list = 1;
%     pList.tau_list = [0.2];
    param.tau_list = [0.6,0.4:-0.09:0.04,0.03,0.02,0.01, 0 ];
    param.tau_list2 = [0.3:-0.05:0.15, 0.12,0.1, 0.09, 0.07,0.04, 0];
    param.tau_list3 = [0.3,0.25,0.2,0.15,0.12, 0.1, 0.08, 0.04, 0.02, 0];
    param.tau_list4 = [0.3:-0.05:0.15, 0.12, 0.1, 0.09, 0.06,  0.03, 0];
    param.lambda_list = 1;
%     pList.lambda_list2     = 10.^(1:0.2:4);
    param.lambda_list2 = 1;
    param.tensor_shape = [4,4,4,4,4,4];% the dimension for dataset dependent reshaped tensors
    param.classsize_list = [20];
    param.nAbl = [0,0,0,0,0,0];
%                              1,0,0,0;
%                              0,1,0,0;
%                              0,0,1,0;
%                              0,0,0,1;
%                              1,1,1,1;
%                              2,2,2,2;
%                              3,3,3,3;
%                              4,4,4,4;];
elseif strcmp(Dataset,'MNIST')
    % MNIST data
    load MNIST.mat
    f = 1;
    if f~=1
        Data_old = Data;
        Data = nan(size(Data,1)*f, size(Data,2)*f, size(Data,3), size(Data,4));
        for i = 1:size(Data,3)
            for j= 1: size(Data, 4)
                Data(:,:, i, j) = imresize(Data_old(:,:,i,j), f);
            end
        end
    end
    img_size    = size(Data);
    images  = reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]);
    labels  = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class           = img_size(ndims(Data));
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list          = 1;
    param.tau_list        = [0.7,0.6:-0.08:0.12,0.03,0.01,0];    %threshold parameter for TTNPE
    param.tau_list2        = [0.48:-0.02:0.28];               	%threshold parameter for 2WTT
    param.tau_list3        = [0.9:-0.1:0.5,0.4:-0.09:0.04,0.03];   %threshold parameter for 3WTT
    param.lambda_list     = 1;
%     pList.lambda_list2     = 10.^(1:0.2:4);
    param.lambda_list2    = 1;
    param.tensor_shape    = [4,7,4,7];% the dimension for dataset dependent reshaped tensors
    param.classsize_list  = [100];
elseif strcmp(Dataset,'COIL3D')
    % COIL with size 64 by 64 by 8 where last mode corresponds to images
    % with 45 degree separation at consecutive indices.
    load COIL3Ddata.mat
    f = 1;
    if f~=1
        Data_old = Data;
        Data = nan(size(Data,1)*f, size(Data,2)*f, size(Data,3), size(Data,4));
        for i = 1:size(Data,3)
            for j= 1: size(Data, 4)
                Data(:,:, i, j) = imresize(Data_old(:,:,i,j), f);
            end
        end
    end
    img_size    = size(Data);
    images  = reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]);
    labels  = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class           = img_size(ndims(Data));
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list          = 1;
    param.tau_list        = [0.5:-0.05:0.15,0]; %threshold parameter for TTNPE
    param.tau_list2        = [0.3:-0.04:0.02,0]; %threshold parameter for TTLDA
    param.tau_list3        = [0.3:-0.04:0.02,0]; %threshold parameter for TTLDA
    param.tau_list4        = [0.3:-0.04:0.02,0];
    param.lambda_list     = [1,1,1];
    param.lambda_list2     = 1;
    param.tensor_shape    = [64,64,8];% the dimension for dataset dependent reshaped tensors
    param.classsize_list  = [3];
    param.nAbl            = [0,0,0];
elseif strcmp(Dataset,'Weizmann')
    % Weizmann Facebase
    load WeizmannData.mat
    img_size        = size(Data);
    images          = double(reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]));
    labels          = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class           = img_size(ndims(Data));
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list          = 1;
    param.tau_list        = [0.16,0.156, .155, .152,0.15:-0.03:0.06,(0.05:-0.02:0.01)/2]*2;
    param.tau_list2       = [0.16,0.156, .155, .152,0.15:-0.03:0.06,(0.05:-0.02:0.01)/2]*2;
    param.tau_list3       = [0.16,0.156, .155, .152,0.15:-0.03:0.06,0.05:-0.015:0.02];
    param.tau_list4       = [0.16,0.156, .155, .152,0.15:-0.03:0.06,0.05:-0.015:0.02];  %threshold parameter for 3WTT
    param.lambda_list     = [1e3];
    param.lambda_list2    = 10.^[0];
    param.tensor_shape    = [8,8,44];% the dimension for dataset dependent reshaped tensors
    param.classsize_list  = [20];
    param.nAbl            = [0,0,0,0,0];
elseif strcmp(Dataset,'UCF101')
    % UCF 101 Sports Action Dataset
    if isunix ==1
        load('/egr/research/sigimprg/Emre/databases/ucf101/UCF-101/matFls/UCF101Data.mat','Data')
    else
        load('\\cifs.egr.msu.edu\research\sigimprg\Emre\databases\ucf101\UCF-101\matFls\UCF101Data.mat','Data')
    end
    img_size    = size(Data);
    images  = reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]);
    labels  = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class           = img_size(ndims(Data));
    param.classsize_list  = [10:10:90];
%     pList.classsize_list  = [60];
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list          = 1;
%     pList.tau_list        = [0.29, .28, 0.24:-0.04:0.04,0.03:-0.02:0.01];
%     pList.tau_list3       = [0.28:-0.04:0.2,0.12,0.03:-0.02:0.01];
%     pList.tau_list4       = [0.28:-0.04:0.2,0.12,0.03:-0.02:0.01];   %threshold parameter for 3WTT
    param.tau_list        = [0.01];
    param.tau_list3        = [0.01];
    param.tau_list4       = [0.12];   %threshold parameter for 3WTT
    param.tensor_shape    = [5,6,5,8,5,10];% the dimension for dataset dependent reshaped tensors
    param.nAbl            = [0,0,0,0,0,0];
elseif strcmp(Dataset,'Cambridge')
    % Cambridge Hand Gesture
    if isunix==1
        load('/egr/research/sigimprg/alp/codes/multiscale_hosvd/classificationrelated/handGestures.mat','Data')
    else
        load('\\cifs.egr.msu.edu\research\sigimprg\alp\codes\multiscale_hosvd\classificationrelated\handGestures.mat','Data')
    end
    img_size    = size(Data);
    images  = double(reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]))/255;
    labels  = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class           = img_size(ndims(Data));
    param.classsize_list  = [4];
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list          = 1;
%     pList.tau_list        = [0.3,0.03:-0.01:0.02 ]; %threshold parameter for TTNPE
%     pList.tau_list2        = [0.3,0.05:-0.01:0.04 ]; %threshold parameter for TTLDA
    param.tau_list        = [0.28:-0.04:0.20,0.12,0.08,0.06,0.04,0.03:-0.01:0.01];
    param.tau_list2       = [0.28:-0.04:0.20,0.12,0.08,0.06,0.04,0.03:-0.01:0.01];
    param.tau_list3       = [0.28,0.24,0.16:-0.04:0.04,0.03:-0.01:0];   %threshold parameter for 3WTT
    param.tensor_shape    = [30,40,30,10];% the dimension for dataset dependent reshaped tensors
    param.nAbl            = [0,0,0,0];
elseif strcmp(Dataset,'KTH')
    % KTH Human Action
    % Deprecated
    load('\\cifs.egr.msu.edu\research\sigimprg\Emre\databases\KTH_Human_Action\KTHData.mat','Data')
    img_size    = size(Data);
    images  = double(reshape(Data, [prod(img_size(1:ndims(Data)-2)),prod(img_size(ndims(Data)-1:end))]))/255;
    labels  = kron(1:img_size(ndims(Data)), ones(1, img_size(ndims(Data)-1)));
    param.class           = img_size(ndims(Data));
    param.classsize_list  = [190];
    clear img_size f i j namelists Data_old Data images_all labels_all;
    param.K_list          = 1;
%     pList.tau_list        = [0.3,0.03:-0.01:0.02 ]; %threshold parameter for TTNPE
%     pList.tau_list2        = [0.3,0.05:-0.01:0.04 ]; %threshold parameter for TTLDA
    param.tau_list        = [0.4:-0.04:0.04,0.03:-0.01:0.01];
    param.tau_list2       = [0.3:-0.03:0.06,0.05:-0.01:0.02]/3;
    param.tau_list3       = [0.4:-0.04:0.04,0.03:-0.01:0.01]/4;   %threshold parameter for 3WTT
    param.lambda_list     = 1;
    param.lambda_list2    = 10.^[0];
    param.tensor_shape    = [30,40,30];% the dimension for dataset dependent reshaped tensors
else
    error('I do not know any such data!');
end
end