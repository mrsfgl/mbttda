function [images, labels,rank_list,tensor_shape,img_shape,class,classsize] = load_dataMDA(Dataset, wR)
% (Input)   Dataset:        a string a a dataset
% (Output)  images:         d * N   vectorized dataset
%           labels:         1 * N   labels for each data
%           K_list          1 * k1 list, each entry is the amount of neibors
%                           to build the graph and conduct KNN classification
%           tau_list        1 * k2 list, each entry is the thresholding parameter for TTNPE
%           tensor_shape    the dimension for dataset dependent reshaped tensors
%           classsize_list  a list of scaler, and each is the amount of data from each class
%           class           a scaler that is the amount of class;

    % COIL
    
if strcmp(Dataset, 'COIL3D')
    load COIL3Ddata.mat
    img_size    = size(Data);
    n = length(img_size)-2;
    images  = reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]);
    labels  = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    img_shape=img_size(1:n);
    tensor_shape    = [64,64,8];% the dimension for dataset dependent reshaped tensors
    tau_list  = [0.3,0.2:-0.04:0.04,0.03:-0.02:0.01,0 ];
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
        for i=1:7
            rank_list(:,i) = (tensor_shape'./8)*(i+1); %rank parameter for DGTDA
        end
    end
    classsize       = [3];
elseif strcmp(Dataset,'COIL')
    % COIL
    load CoilData.mat
    tensor_shape    = [8,8,8,8];% the dimension for dataset dependent reshaped tensors
    Data=reshape(Data,[tensor_shape,size(Data,3),size(Data,4)]);
    img_size    = size(Data);
    n = length(img_size)-2;
    images  = reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]);
    labels  = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    img_shape=img_size(1:n);
    tau_list  = [0.95,0.9:-0.2:0.5,0.3,0.12,0.04,0.03:-0.02:0.01,0 ];
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
        for i=1:7
            rank_list(:,i) = (tensor_shape'./8)*(i+1); %rank parameter for DGTDA
        end
    end
    classsize       = [20];
elseif strcmp(Dataset,'MNIST')
    % Weizman
    load MNIST.mat
    tensor_shape    = [4,7,4,7];% the dimension for dataset dependent reshaped tensors
    Data=reshape(Data,[tensor_shape,size(Data,3),size(Data,4)]);
    img_size    = size(Data);
    n = length(img_size)-2;
    images  = reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]);
    labels  = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    img_shape=img_size(1:n);
    tau_list  = [0.6:-0.1:0.5,0.4,0.3,0.2:-0.08:0.04,0.03:-0.01:0.01,0 ];
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
        for i=1:7
            rank_list(:,i) = (tensor_shape'./8)*(i+1); %rank parameter for DGTDA
        end
    end
    classsize  = [100];
elseif strcmp(Dataset, 'GAIT')
    % Finance
    load GaitData.mat
    tensor_shape    = [10,6,11,8,10,5];% the dimension for dataset dependent reshaped tensors
    Data            = reshape(Data,[tensor_shape,size(Data,ndims(Data)-1),size(Data,ndims(Data))]);
    img_size        = size(Data);
    n               = length(img_size)-2;
    img_shape       = img_size(1:n-2);
    images          = double(reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]))/255;
    labels          = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    tau_list        = [0.24:-0.03:0.06,0.05:-0.015:0.02]*2;
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
%             for i=1:4
%                 rank_list(:,i) = floor((tensor_shape'./5))*(i+1); %rank parameter for DGTDA
%             end
            rank_list = [1,2,4,8,16,32,64;1,2,4,6,10,16,44;1,2,2,3,3,4,5];
    end
    clear img_size f i j namelists Data_old Data images_all labels_all C
    classsize  = [38];
elseif strcmp(Dataset,'YaleB')
    % MNIST
    load YaleBData.mat
    Data = permute(Data,[1,2,5,4,3]);
    tensor_shape    = [30,40,20];% the dimension for dataset dependent reshaped tensors
    Data            = reshape(Data,[tensor_shape,size(Data,ndims(Data)-1),size(Data,ndims(Data))]);
    img_size        = size(Data);
    n               = length(img_size)-2;
    img_shape       = img_size(1:n-2);
    images          = double(reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]))/255;
    labels          = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    tau_list        = [0.24:-0.03:0.06,0.05:-0.015:0.02]*2;
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
%             for i=1:4
%                 rank_list(:,i) = floor((tensor_shape'./5))*(i+1); %rank parameter for DGTDA
%             end
            rank_list = [1,2,4,8,16,32,64;1,2,4,6,10,16,44;1,2,2,3,3,4,5];
    end
    clear img_size f i j namelists Data_old Data images_all labels_all C
    classsize  = [32];
elseif strcmp(Dataset,'Weizmann')
    % Weizman
    load WeizmannData.mat
    tensor_shape    = [4,4,4,4,11];% the dimension for dataset dependent reshaped tensors
    Data            = reshape(Data,[tensor_shape,size(Data,ndims(Data)-1),size(Data,ndims(Data))]);
    img_size        = size(Data);
    n               = length(img_size)-2;
    img_shape       = img_size(1:n-2);
    images          = double(reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]))/255;
    labels          = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    tau_list        = [0.24,0.18:-0.03:0.06,0.05:-0.015:0.005];
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
%             for i=1:4
%                 rank_list(:,i) = floor((tensor_shape'./5))*(i+1); %rank parameter for DGTDA
%             end
            rank_list = [1,2,4,8,16,32,64;1,2,4,6,10,16,44;1,2,2,3,3,4,5];
    end
    clear img_size f i j namelists Data_old Data images_all labels_all C
    tensor_shape    = [4,4,4,4,11];% the dimension for dataset dependent reshaped tensors
    classsize  = [20];
elseif strcmp(Dataset,'Cambridge')
    % Weizman
    load('\\cifs.egr.msu.edu\research\sigimprg\alp\codes\multiscale_hosvd\classificationrelated\handGestures.mat','Data')
    img_size    = size(Data);
    n = length(img_size)-2;
    img_shape=img_size(1:ndims(Data)-2);
    images  = double(reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]))/255;
    labels  = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    tau_list  = [0.3,0.2:-0.08:0.04,0.03:-0.01:0.01,0 ];
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
%             for i=1:4
%                 rank_list(:,i) = floor((tensor_shape'./5))*(i+1); %rank parameter for DGTDA
%             end
        rank_list = [2,3,6,15,20;2,4,8,20,30;2,3,6,15,20;2,3,4,5,6];
    end
    tensor_shape    = [30,40,30,10];% the dimension for dataset dependent reshaped tensors
    classsize  = [4];
elseif strcmp(Dataset,'KTH')
    % Weizman
    load('\\cifs.egr.msu.edu\research\sigimprg\Emre\databases\KTH_Human_Action\KTHData.mat','Data')
    img_size    = size(Data);
    n = length(img_size)-2;
    img_shape=img_size(1:ndims(Data)-2);
    images  = double(reshape(Data, [prod(img_size(1:n)),prod(img_size(n+1:end))]))/255;
    labels  = kron(1:img_size(n+2), ones(1, img_size(n+1)));
    class           = img_size(n+2);
    tau_list  = [0.09,0.06,0.04:-0.005:0.01,0 ];
    if ~wR
        for i=1:length(tau_list)
            C=hosvd2(double(Data),[ones(1,n),0,0],tau_list(i));
            s=size(C);
            rank_list(:,i)=s(1:n);
        end
    else
%             for i=1:4
%                 rank_list(:,i) = floor((tensor_shape'./5))*(i+1); %rank parameter for DGTDA
%             end
        rank_list = [2,3,6,15,20;2,4,8,20,30;2,3,6,15,20;2,3,4,5,6];
    end
    tensor_shape    = [30,40,30];% the dimension for dataset dependent reshaped tensors
    classsize  = [190];
else
    error('Wrong Name!!');
end
end