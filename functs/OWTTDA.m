function OWTTDA = OWTTDA(train_data, train_label, test_data,test_label, I, tau, param,KNN)
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
n   = length(I);     % tensor data order
N   = size(train_data, 2);      % train data size
cSize = length(find(train_label==train_label(1)));
numC = N/cSize;

train_data=reshape(train_data, [I,cSize,numC]);
test_data=reshape(test_data, [I,size(test_data,2)/numC,numC]);
[I,sI]=sort(I,'descend');
train_data=permute(train_data,[sI,n+1,n+2]);
train_data=reshape(train_data, [prod(I), cSize,numC]);
test_data=reshape(permute(test_data,[sI,n+1,n+2]),prod(I),[]);
% Initialize Ui
Ui = U2Ui_tau(reshape(train_data, [I, N]), tau); % U_1, ....,U_k, A 
Ui = Ui(1:end-1); 
r=size(Ui{end},3);


%% Loop
for iter =1:param.maxiterL
    % Learn Subspaces
    [Ui,matTime,searchTime] = lrnU(train_data,Ui,param,opts,cSize,numC);
    Sub  = reshape(merge_tensor(Ui),[],r);
    OWTTDA.Ui(iter,1:n)=Ui;
    
    % Projections
    TTLDA_start = cputime;
    mt = cputime;
    train_proj=reshape(tmprod(train_data,Sub',1),[r,N]);
    mt = cputime - mt;
    OWTTDA.Ui{iter,n+1}=train_proj;
    test_proj=tmprod(test_data,Sub',1);
    
    OWTTDA.time_embedding(iter) = cputime - TTLDA_start;   % time to embed data
    OWTTDA.time_mat(iter) = matTime + mt;
    OWTTDA.time_search(iter) = searchTime;
    OWTTDA.time_subspace(iter) = OWTTDA.time_mat(iter) + OWTTDA.time_search(iter);
    
    % Classifications
    
    TTLDA_start = cputime;
    [OWTTDA.PreLabel{iter}, OWTTDA.PreErr(:,iter)] = Classfier_KNN(train_proj, train_label, test_proj, test_label, KNN.K);
    OWTTDA.time_classify(iter) = cputime - TTLDA_start;   % time to classify data
    OWTTDA.Storage(iter)  = Dim_TT(Ui(1:n))+numel(OWTTDA.Ui{iter,n+1});
    %% End of Loop
end

end