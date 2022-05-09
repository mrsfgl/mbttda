function TTDA = ttda(data, param)
%% TTDA = ttda(data, param)
%  Tensor Train Discriminant Analysis. 
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
%
%  Seyyid Emre Sofuoglu

% unitary solver parameter
opts.record = 0;
opts.mxitr  = 1000;
opts.gtol   = 1e-5;
opts.xtol   = 1e-5;
opts.ftol   = 1e-8;
opts.param.tau    = 1e-3;

% read parameter
n   = length(param.I);     % tensor data order
N   = size(data.train, 2);      % train data size
cSize = length(find(data.trLbl==data.trLbl(1)));
numC = N/cSize;

data.train=reshape(data.train, [param.I,cSize,numC]);
data.test=reshape(data.test, [param.I,size(data.test,2)/numC,numC]);
[param.I,sI]=sort(param.I,'descend');
data.train=permute(data.train,[sI,n+1,n+2]);
data.train=reshape(data.train, [prod(param.I), cSize,numC]);
data.test=reshape(permute(data.test,[sI,n+1,n+2]),prod(param.I),[]);
% Initialize Ui
Ui = U2Ui_tau(reshape(data.train, [param.I, N]), param.tau); % U_1, ....,U_k, A 
Ui = Ui(1:end-1); 
r=size(Ui{end},3);
TTDA.objVal = [];

%% Loop
for iter =1:param.maxiterOut
    % Learn Subspaces
    [Ui,searchTime, objVal] = lrnU(data.train,Ui,param,opts,cSize,numC);
    TTDA.objVal = [TTDA.objVal, objVal];
    for i = 1:n
        ablInd = randi(param.I(i), param.nA(i), 1);
        Ui{i}(:, ablInd, :) = 0;
    end
    Sub  = reshape(merge_tensor(Ui),[],r);
    TTDA.Ui(1:n)=Ui;
    
    % Projections
    TTLDA_start = cputime;
    train_proj=reshape(tmprod(data.train,Sub',1),[r,N]);
    TTDA.Ui{n+1}=train_proj;
    test_proj=tmprod(data.test,Sub',1);
    
    TTDA.time_embedding(iter) = cputime - TTLDA_start;   % time to embed data
    TTDA.time_search(iter) = searchTime;
    TTDA.time_subspace(iter) = TTDA.time_search(iter);
    
    % Classifications
    
    TTLDA_start = cputime;
    [TTDA.PreLabel{iter}, TTDA.PreErr(:,iter)] = Classfier_KNN(train_proj, data.trLbl, test_proj, data.tsLbl, 1);
    TTDA.time_classify(iter) = cputime - TTLDA_start;   % time to classify data
%     TTDA.Storage(iter)  = Dim_TT(Ui(1:n))+numel(TTDA.Ui{iter,n+1});
    %% End of Loop
end
S1 = whos('Ui');
S2 = whos('train_proj');
TTDA.Ui = [];
Ui = [];
tmp= data.train;
S3 = whos('tmp');
TTDA.Storage  = (S1.bytes+S2.bytes)/S3.bytes;

end