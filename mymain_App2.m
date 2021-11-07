function TTDA = mymain_App2(train_data, train_label, test_data,test_label, I, tau, para_App,KNN)
% train_data  & test_data:  (I1*...*In) * N matrix
% train_label & test_label: 1 * N

% unitary solver parameter
opts.record = 0;
opts.mxitr  = 1000;
opts.gtol   = 1e-5;
opts.xtol   = 1e-5;
opts.ftol   = 1e-8;
opts.tau    = 1e-3;

% read parameter
k   = computeK(I,2);
n   = length(I);     % tensor data order
N   = size(train_data, 2);      % train data size
cSize = length(find(train_label==train_label(1)));
numC = N/cSize;
Ls=prod(I(1:k));
Rs=prod(I(k+1:n));
mD{1}=1:k;
mD{2}=k+1:n;


train_data=reshape(train_data,[I,N]);
test_data=reshape(test_data,[I,size(test_data,2)]);
% [I(1:k),mD{1}]=sort(I(1:k),'descend');
% [I(k+1:n),mD{2}]=sort(I(k+1:n),'descend'); mD{2}=mD{2}+k;
train_data=permute(train_data,[mD{1},n+1,mD{2}]);
train_data=reshape(train_data, [Ls,cSize,numC,Rs]);
% Initialize Ui
Ui = U2Ui_tau(reshape(train_data, [I(1:k), N*Rs]), tau); % U_1, ....,U_k, A 
Ui = Ui(1:k); 
rln=size(Ui{k},3);
LSub         = reshape(merge_tensor(Ui(1:k)),[],rln);
lprojTrain=permute(tmprod(train_data,LSub',1),[4,2,3,1]);
Ui(k+1:n+1) = U2Ui_tau(reshape(lprojTrain,[I(k+1:n),N*rln]), tau); % U_1, ....,U_k, A 
rrn=size(Ui{n},3);

test_data=permute(test_data,[mD{1},mD{2},n+1]);
test_data=permute(reshape(test_data,Ls,Rs,[]),[1,3,2]);



%% Loop
rprojTrain=train_data;
for iter =1:para_App.maxiterL
    % Left operations
    [Ui(1:k),Lmattime,Lsearchtime] = lrnU(rprojTrain,Ui(1:k),para_App,opts);
    LSub         = reshape(merge_tensor(Ui(1:k)),[],rln);
    TTDA.Ui(iter,1:k)=Ui(1:k);
    lmtt=0;
    % Right Operations
    rmtt = cputime;
    lprojTrain=permute(tmprod(train_data,LSub',1),[4,2,3,1]);
    rmtt = cputime-rmtt;
    [Ui(k+1:n),Rmattime,Rsearchtime] = lrnU(lprojTrain,Ui(k+1:n),para_App,opts);
    RSub         = reshape(merge_tensor(Ui(k+1:n)),[],rrn);
    
    TTDA.Ui(iter,k+2:n+1) = Ui(n:-1:k+1);
    for i=(k+2:n+1)
        TTDA.Ui{iter,i}=permute(TTDA.Ui{iter,i},[3,2,1]);
    end
    
    % Projections
    TTLDA_start = cputime;
    lmtt = cputime;
    rprojTrain=tmprod(train_data,RSub',4);
    lmtt = cputime - lmtt;
    TTDA.Ui{iter,k+1}=reshape(tmprod(rprojTrain,LSub',1),[rln,N,rrn]);
    train_proj=ndim_unfold(TTDA.Ui{iter,k+1},2)';
    test_proj=ndim_unfold(tmprod(tmprod(test_data,LSub',1),RSub',3),2)';
    
    TTDA.time_embedding(iter) = cputime - TTLDA_start;   % time to embed data
    TTDA.time_mat(iter) = Lmattime + Rmattime + rmtt + lmtt;
    TTDA.time_search(iter) = Lsearchtime + Rsearchtime;
    TTDA.time_subspace(iter) = TTDA.time_mat(iter) + TTDA.time_search(iter);
    
    % Classifications
    
    TTLDA_start = cputime;
    [TTDA.PreLabel{iter}, TTDA.PreErr(:,iter)] = Classfier_KNN(train_proj, train_label, test_proj, test_label, KNN.K);
    TTDA.time_classify(iter) = cputime - TTLDA_start;   % time to classify data
    TTDA.Storage(iter)  = Dim_TT(Ui(1:n)) + numel(TTDA.Ui{iter,k+1});
    %% End of Loop
end

end