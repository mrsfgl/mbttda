function BTT = btt(data, param)
%% btt = btt(data, par)
%  Block Tensor Train Discriminant Analysis. 
%  -------------------------
%  data: Struct that includes training and test data matrices and
%  corresponding labels.
%  
%  param : Struct of parameters.
%    param.I: tensor shape of each sample:
%    param.tau: Threshold parameter tau for TT-PCA.
%    param.rndI: random initialization struct:
%         param.rndI.flag: if true, initializes U's randomly. No TTPCA
%         param.rndI.rank: needs to have elements for each tensor mode
%         indicating the ranks of tensor factors if flag is true.
%    param.maxIterOut: Outer loop max iteration.
%    param.nA: Ablation indices for each mode. Sets that slice to zero.
%    param.lambda: Parameter controlling the balance of within and
%    between class scatters.
%    param.display: Displays and extracts objective value.
%    param.maxiter: Inner loop max iteration.
%    param.error_tot: Inner loop error threshold.
%  --------------------------
% This function uses Tensor-Train Toolbox kindly
% provided by [*]
%
%  Seyyid Emre Sofuoglu
% 
% [*] Dolgov, Khoromskij, Oseledets, Savostyanov,
% "Computation of extreme eigenvalues in higher dimensions using block
% tensor train format", Comp. Phys. Comm. 2014, http://dx.doi.org/10.1016/j.cpc.2013.12.017
%

%% Pre-process
n = length(param.I);     % tensor data order
num_tr_samp = size(data.train, 2);      % train data size
class_size = length(find(data.trLbl == data.trLbl(1)));
num_class = num_tr_samp/class_size;

data.train = reshape(data.train, [prod(param.I), class_size,num_class]);
%% Initialize Ui
Ui = U2Ui_tau(reshape(data.train, [param.I, num_tr_samp]), param.tau); % U_1, ....,U_k, A 
Ui = Ui(1:end-1); 
r = size(Ui{end},3);

%% Loop
for iter =1:param.maxiterOut
    % Learn Subspaces
    [Ui, train_time] = lrnU_btt(data.train, Ui, param);
%     for i = 1:n
%         ablInd = randi(param.I(i), param.nA(i), 1);
%         Ui{i}(:, ablInd, :) = 0;
%     end
    Sub = reshape(merge_tensor(Ui),[],r);
    for i = 1:n
        BTT.Ui{i} = full(Ui{i});
    end
    
    % Projections
    btt_start = cputime;
    train_proj = reshape(tmprod(data.train,Sub',1),[r,num_tr_samp]);
    BTT.Ui{n+1} = train_proj;
    test_proj = Sub'*data.test;
    
    BTT.time_embedding(iter) = cputime - btt_start;   % time to embed data
    BTT.time_search(iter) = train_time;
    BTT.time_subspace(iter) = BTT.time_search(iter);
    
    % Classifications
    btt_start = cputime;
    [BTT.PreLabel{iter}, BTT.PreErr(:,iter)] = Classfier_KNN(train_proj, data.trLbl, test_proj, data.tsLbl, 1);
    BTT.time_classify(iter) = cputime - btt_start;   % time to classify data
    %% End of Loop
end
S1 = whos('Ui');
S2 = whos('train_proj');
clear Ui BTT.Ui
tmp = data.train;
S3 = whos('tmp');
BTT.Storage = (S1.bytes+S2.bytes)/S3.bytes;

end