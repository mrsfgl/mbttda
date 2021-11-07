function cmda = main_CMDA(train_data, train_label, test_data,test_label, I, ranks, KNN, patches)
% train_data  & test_data:  (I1*...*In) * N matrix
% train_label & test_label: 1 * N

% read parameter
n   = length(I);     % tensor data order
cSize = length(find(train_label==train_label(1)));
N   = size(train_data, ndims(train_data)) * cSize;     % train data size
tSize = size(test_data, ndims(test_data)) * size(test_data, ndims(test_data)-1) ;

if ~patches
    N = N/ cSize;
    numC = N/ cSize;
    train_data=reshape(train_data,[I,cSize,numC]);
    test_data=reshape(test_data,[I,size(test_data,2)]);
    tSize = size(test_data,n+1);
end

cmda.time_subspace = cputime;
cmda.Ui=CMDA(train_data,[],ranks);
cmda.time_subspace = cputime-cmda.time_subspace;    % time to find subspace

% Projections
cmda.time_embedding = cputime;
train_proj = tmprod(train_data,cellfun(@transpose,cmda.Ui,'UniformOutput',false),1:length(cmda.Ui));
test_proj  = tmprod(test_data,cellfun(@transpose,cmda.Ui,'UniformOutput',false),1:length(cmda.Ui));
cmda.time_embedding = cputime-cmda.time_embedding;

train_proj=reshape(train_proj,[prod([ranks,I(length(ranks)+1:end)]),N]);
test_proj=reshape(test_proj,[prod([ranks,I(length(ranks)+1:end)]),tSize]);
cdast = cputime;
[cmda.PreLabel, cmda.PreErr] = Classfier_KNN(train_proj, train_label, test_proj, test_label, KNN.K);
cmda.time_classify = cputime - cdast;   % time to classify data
cmda.Storage  = sum(ranks.*I(1:length(ranks)))+prod([ranks,I(length(ranks)+1:end)])*N;

end