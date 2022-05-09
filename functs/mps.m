function MPS = mps(data, par)
%% MPS = mps(data, par)
%  Matrix Product State/TT decomposition. TTPCA is applied.
%  -------------------------
%  data: Struct that includes training and test data matrices and
%  corresponding labels.
%  
%  par : Struct of parameters.
%    par.I: tensor shape of each sample:
%    par.tau: Threshold parameter tau for TT-PCA.
%    par.rndI: random initialization struct:
%         par.rndI.flag: if true, initializes U's randomly. No TTPCA
%         par.rndI.rank: needs to have elements for each tensor mode
%         indicating the ranks of tensor factors if flag is true.
%  --------------------------
%
%  Seyyid Emre Sofuoglu


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


data.train=reshape(data.train,[par.I,N]);
data.test=reshape(data.test,[par.I,size(data.test,2)]);
% [par.I(1:k),mD{1}]=sort(par.I(1:k),'descend');
% [par.I(k+1:n),mD{2}]=sort(par.I(k+1:n),'descend'); mD{2}=mD{2}+k;
data.train=permute(data.train,[mD{1},n+1,mD{2}]);
data.train=reshape(data.train, [Ls,cSize,numC,Rs]);

% Initialize Ui
tic
Ui = U2Ui_tau(reshape(data.train, [par.I(1:k), N*Rs]), par.tau, par.rndI); % U_1, ....,U_k, A 
Ui = Ui(1:k); 
rln=size(Ui{k},3);
LSub         = reshape(merge_tensor(Ui(1:k)),[],rln);
lprojTrain   = permute(tmprod(data.train,LSub',1),[4,2,3,1]);
Ui(k+1:n+1)  = U2Ui_tau(reshape(lprojTrain,[par.I(k+1:n),N*rln]), par.tau, par.rndI); % U_1, ....,U_k, A 
rrn=size(Ui{n},3);
RSub         = reshape(merge_tensor(Ui(k+1:n)),[],rrn);
MPS.time_subspace = toc;

data.test=permute(data.test,[mD{1},mD{2},n+1]);
data.test=permute(reshape(data.test,Ls,Rs,[]),[1,3,2]);

Ui{n+1} = permute(reshape(Ui{n+1}, rrn, N, rln),[3,2,1]);
train_proj = ndim_unfold(Ui{n+1},2)';
test_proj  = ndim_unfold(tmprod(tmprod(data.test,LSub',1),RSub',3),2)';

% Classifications    
[MPS.PreLabel, MPS.PreErr] = Classfier_KNN(train_proj, data.trLbl, test_proj, data.tsLbl, 1);

S1 = whos('Ui');
tmp= data.train;
S3 = whos('tmp');
MPS.Storage  = (S1.bytes)/S3.bytes;

end