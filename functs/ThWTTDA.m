function ThW = thwttda(data, param)
%% ThW = thwttda(data, param)
%  3-Way Tensor Train Discriminant Analysis. 
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
k   = computeK(param.I, 3);

n   = length(param.I);     % tensor data order
N   = size(data.train, 2);      % train data size
cSize = length(find(data.trLbl==data.trLbl(1)));
numC = N/cSize;
Ls=prod(param.I(1:k(1)));
Ms=prod(param.I(k(1)+1:k(2)));
Rs=prod(param.I(k(2)+1:n));
mD{1}=1:k(1);
mD{2}=k(1)+1:k(2);
mD{3}=k(2)+1:n;


data.train=reshape(data.train,[param.I,N]);
data.test=reshape(data.test,[param.I,size(data.test,2)]);
% [param.I(1:k(1)),mD{1}]=sort(param.I(1:k(1)),'descend');
% [param.I(k(1)+1:k(2)),mD{2}]=sort(param.I(k(1)+1:k(2)),'descend');mD{2}=mD{2}+k(1);
% [param.I(k(2)+1:n),mD{3}]=sort(param.I(k(2)+1:n),'descend');mD{3}=mD{3}+k(2);
data.train=permute(data.train,[mD{1},n+1,mD{2},mD{3}]);
data.train=reshape(data.train, [Ls,cSize,numC,Ms,Rs]);

% Initialize Ui
Ui = U2Ui_tau(reshape(data.train, [param.I(1:k(1)), N*Rs*Ms]), param.tau, param.rndI); % U_1, ....,U_m
Ui = Ui(1:k(1)); 
rln = size(Ui{k(1)},3);
LSub = reshape(merge_tensor(Ui(1:k(1))),[],rln);
lprojdTrain=permute(tmprod(data.train,LSub',1),[4,2,3,5,1]);
Ui(k(1)+1:k(2)+1) = U2Ui_tau(reshape(lprojdTrain,[param.I(k(1)+1:k(2)),N*Rs*rln]), param.tau, param.rndI); % U_{m+1}, ....,U_n
Ui = Ui(1:k(2));
rmn=size(Ui{k(2)},3);
MSub = reshape(merge_tensor(Ui(k(1)+1:k(2))),[],rmn);
mprojdTrain=permute(tmprod(lprojdTrain,MSub',1),[4,2,3,5,1]);
Ui(k(2)+1:n+1) = U2Ui_tau(reshape(mprojdTrain,[param.I(k(2)+1:n),N*rln*rmn]), param.tau, param.rndI); % U_{m+1}, ....,U_n
Ui = Ui(1:n);
rrn=size(Ui{n},3);
RSub = reshape(merge_tensor(Ui(k(2)+1:n)),[],rrn);

data.test=permute(data.test,[mD{1},mD{2},mD{3},n+1]);
data.test=permute(reshape(data.test,Ls,Ms,Rs,[]),[1,4,2,3]);

ThW.objVal = [];


%% Loop
rprojTrain = reshape(tmprod(data.train,{MSub',RSub'},[4,5]),[Ls,cSize,numC,rmn*rrn]);
for iter =1:param.maxiterOut
    % Left operations
    [Ui(1:k(1)), l_sub_time, tmp] = lrnU(rprojTrain,Ui(1:k(1)),param,opts);
    ThW.objVal = [ThW.objVal, tmp];
    LSub  = reshape(merge_tensor(Ui(1:k(1))),[],rln);
    ThW.Ui(1:k(1))=Ui(1:k(1));
    % Middle Operations
    rmtt = cputime;
    lprojdTrain=reshape(permute(tmprod(data.train,{LSub',RSub'},[1,5]),[4,2,3,5,1]),[Ms,cSize,numC,rrn*rln]);
    rmtt = cputime-rmtt;
    [Ui(k(1)+1:k(2)), m_sub_time, tmp] = lrnU(lprojdTrain,Ui(k(1)+1:k(2)),param,opts);
    ThW.objVal = [ThW.objVal, tmp];
    MSub  = reshape(merge_tensor(Ui(k(1)+1:k(2))),[],rmn);
    ThW.Ui(k(1)+2:k(2)+1) = Ui(k(2):-1:k(1)+1);
    for i=(k(1)+2:k(2)+1)
        ThW.Ui{i}=permute(ThW.Ui{i},[3,2,1]);
    end
    % Right Operations
    mmtt = cputime;
    mprojdTrain=reshape(permute(tmprod(data.train,{LSub',MSub'},[1,4]),[5,2,3,1,4]),[Rs,cSize,numC,rln*rmn]);
    mmtt = cputime-mmtt;
    [Ui(k(2)+1:n), r_sub_time, tmp] = lrnU(mprojdTrain,Ui(k(2)+1:n),param,opts);
    ThW.objVal = [ThW.objVal, tmp];
    RSub  = reshape(merge_tensor(Ui(k(2)+1:n)),[],rrn);
    
    ThW.Ui(k(2)+2:n+1) = Ui(n:-1:k(2)+1);
    for i=(k(2)+2:n+1)
        ThW.Ui{i}=permute(ThW.Ui{i},[3,2,1]);
    end
    
    % Projections
    TTLDA_start = cputime;
    lmtt = cputime;
    rprojTrain = reshape(tmprod(data.train,{MSub',RSub'},[4,5]),[Ls,cSize,numC,rmn*rrn]);
    lmtt = cputime - lmtt;
    ThW.Ui{k(1)+1} = reshape(tmprod(rprojTrain,LSub',1),[rln,N,rmn,rrn]);
    train_proj = ndim_unfold(ThW.Ui{k(1)+1},2)';
    
    for i = 1:n
        ablInd = randi(param.I(i), param.nA(i), 1);
        Ui{i}(:, ablInd, :) = 0;
    end
    LSub2 = reshape(merge_tensor(Ui(1:k(1))),[],rln);
    MSub2 = reshape(merge_tensor(Ui(k(1)+1:k(2))),[],rmn);
    RSub2 = reshape(merge_tensor(Ui(k(2)+1:n)),[],rrn);
    test_proj = ndim_unfold(tmprod(data.test,{LSub2',MSub2',RSub2'},[1,3,4]),2)';
    ThW.time_embedding(iter) = cputime - TTLDA_start;   % time to embed data
    
    ThW.time_subspace(iter) = l_sub_time + m_sub_time + r_sub_time;    % time to find subspace
    ThW.time_mat(iter) = rmtt + mmtt + lmtt;
    ThW.time_search(iter) = l_sub_time + m_sub_time + r_sub_time;
    
    TTLDA_start = cputime;
    [ThW.PreLabel{iter}, ThW.PreErr(:,iter)] = Classfier_KNN(train_proj, data.trLbl, test_proj, data.tsLbl, 1);
    ThW.time_classify(iter) = cputime - TTLDA_start;   % time to classify data
    %% End of Loop
end
S1 = whos('Ui');
S2 = whos('train_proj');
Ui = [];
ThW.Ui = [];
tmp = data.train;
S3 = whos('tmp');
ThW.Storage  = (S1.bytes+S2.bytes)/S3.bytes;
end