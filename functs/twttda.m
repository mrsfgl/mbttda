function TTDA = twttda(data, par)
%% TTDA = twttda(data, par)
%  2-Way Tensor Train Discriminant Analysis. 
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
opts.par.tau    = 1e-3;

% read parameter
k   = computeK(par.I,2);
n   = length(par.I);     % tensor data order
N   = size(data.train, 2);      % train data size
cSize = length(find(data.trLbl==data.trLbl(1)));
numC = N/cSize;
Ls=prod(par.I(1:k));
Rs=prod(par.I(k+1:n));
mD{1}=1:k;
mD{2}=k+1:n;


data.train = reshape(data.train,[par.I,N]);
data.test = reshape(data.test,[par.I,size(data.test,2)]);
% [par.I(1:k),mD{1}]=sort(par.I(1:k),'descend');
% [par.I(k+1:n),mD{2}]=sort(par.I(k+1:n),'descend'); mD{2}=mD{2}+k;
data.train = permute(data.train,[mD{1},n+1,mD{2}]);
data.train = reshape(data.train, [Ls,cSize,numC,Rs]);

% Initialize Ui
Ui = U2Ui_tau(reshape(data.train, [par.I(1:k), N*Rs]), par.tau, par.rndI); % U_1, ....,U_k, A 
Ui = Ui(1:k); 
rln = size(Ui{k},3);
LSub = reshape(merge_tensor(Ui(1:k)),[],rln);
lprojTrain = permute(tmprod(data.train,LSub',1),[4,2,3,1]);
Ui(k+1:n+1) = U2Ui_tau(reshape(lprojTrain,[par.I(k+1:n),N*rln]), par.tau, par.rndI); % U_1, ....,U_k, A 
Ui = Ui(1:n);
rrn = size(Ui{n},3);
RSub = reshape(merge_tensor(Ui(k+1:n)),[],rrn);

data.test = permute(data.test,[mD{1},mD{2},n+1]);
data.test = permute(reshape(data.test,Ls,Rs,[]),[1,3,2]);

TTDA.objVal = [];

%% Loop
rprojTrain=tmprod(data.train,RSub',4);
for iter =1:par.maxiterOut
    % Left operations
    [Ui(1:k),Lsearchtime,tmp] = lrnU(rprojTrain,Ui(1:k),par,opts);
    TTDA.objVal = [TTDA.objVal, tmp];
    LSub         = reshape(merge_tensor(Ui(1:k)),[],rln);
    TTDA.Ui(1:k)=Ui(1:k);
    % Right Operations
    lprojTrain=permute(tmprod(data.train,LSub',1),[4,2,3,1]);
    [Ui(k+1:n),Rsearchtime,tmp] = lrnU(lprojTrain,Ui(k+1:n),par,opts);
    TTDA.objVal = [TTDA.objVal, tmp];
    RSub         = reshape(merge_tensor(Ui(k+1:n)),[],rrn);
    
    TTDA.Ui(k+2:n+1) = Ui(n:-1:k+1);
    for i=(k+2:n+1)
        TTDA.Ui{i}=permute(TTDA.Ui{i},[3,2,1]);
    end
    
    % Projections
    tic
    rprojTrain=tmprod(data.train,RSub',4);
    TTDA.Ui{k+1}=reshape(tmprod(rprojTrain,LSub',1),[rln,N,rrn]);
    train_proj=ndim_unfold(TTDA.Ui{k+1},2)';
    for i = 1:n
        ablInd = randi(par.I(i), par.nA(i), 1);
        Ui{i}(:, ablInd, :) = 0;
    end
    LSub2 = reshape(merge_tensor(Ui(1:k)),[],rln);
    RSub2 = reshape(merge_tensor(Ui(k+1:n)),[],rrn);
    test_proj=ndim_unfold(tmprod(tmprod(data.test,LSub2',1),RSub2',3),2)';
    
    TTDA.time_embedding(iter) = toc;   % time to embed data
    TTDA.time_subspace(iter) = Lsearchtime + Rsearchtime;
    
    % Classifications
    
    tic
    [TTDA.PreLabel{iter}, TTDA.PreErr(:,iter)] = Classfier_KNN(train_proj, data.trLbl, test_proj, data.tsLbl, 1);
    TTDA.time_classify(iter) = toc;   % time to classify datUi
    
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