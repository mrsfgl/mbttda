function cmda = main_CMDA(data, I, ranks, KNN, patches)
% data.train  & data.test:  (I1*...*In) * N matrix
% data.trLbl & data.tsLbl: 1 * N

% read parameter
n   = length(I);     % tensor data order
cSize = length(find(data.trLbl==data.trLbl(1)));
N   = size(data.train, ndims(data.train)) * cSize;     % train data size
tSize = size(data.test, ndims(data.test)) * size(data.test, ndims(data.test)-1) ;

if ~patches
    N = N/ cSize;
    numC = N/ cSize;
    data.train=reshape(data.train,[I,cSize,numC]);
    data.test=reshape(data.test,[I,size(data.test,2)]);
    tSize = size(data.test,n+1);
end

cmda.time_subspace = cputime;
cmda.Ui=CMDA(data.train,[],ranks);
cmda.time_subspace = cputime-cmda.time_subspace;    % time to find subspace

% Projections
cmda.time_embedding = cputime;
train_proj = tmprod(data.train,cellfun(@transpose,cmda.Ui,'UniformOutput',false),1:length(cmda.Ui));
test_proj  = tmprod(data.test,cellfun(@transpose,cmda.Ui,'UniformOutput',false),1:length(cmda.Ui));
cmda.time_embedding = cputime-cmda.time_embedding;

train_proj=reshape(train_proj,[prod([ranks,I(length(ranks)+1:end)]),N]);
test_proj=reshape(test_proj,[prod([ranks,I(length(ranks)+1:end)]),tSize]);
cdast = cputime;
[cmda.PreLabel, cmda.PreErr] = Classfier_KNN(train_proj, data.trLbl, test_proj, data.tsLbl, KNN);
cmda.time_classify = cputime - cdast;   % time to classify data
S1 = whos('train_proj');
Ui = cmda.Ui;
S2 = whos('Ui');
cmda.Storage  = (S1.bytes+S2.bytes)/numel(data.train)/8;

end