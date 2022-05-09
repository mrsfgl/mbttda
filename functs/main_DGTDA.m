function dgtda = main_DGTDA(data, I, ranks, KNN, patches)
% data.train  & data.test:  (I1*...*In) * N matrix
% data.trLbl & data.tsLbl: 1 * N

% read parameter
n   = length(I);     % tensor data order
cSize = length(find(data.trLbl==data.trLbl(1)));
N   = size(data.train, ndims(data.train)) * cSize;      % train data size
tSize = size(data.test, ndims(data.test)) * size(data.test, ndims(data.test)-1) ;

if ~patches
    N = N/ cSize;
    numC = N/ cSize;
    data.train=reshape(data.train,[I,cSize,numC]);
    data.test=reshape(data.test,[I,size(data.test,2)]);
    tSize = size(data.test,n+1);
end

dgtda.time_subspace = cputime;
dgtda.Ui=DGTDA(data.train,ranks);
dgtda.time_subspace = cputime-dgtda.time_subspace;    % time to find subspace

% Projections
dgtda.time_embedding = cputime;
train_proj = tmprod(data.train,cellfun(@transpose,dgtda.Ui,'UniformOutput',false),1:length(dgtda.Ui));
test_proj  = tmprod(data.test,cellfun(@transpose,dgtda.Ui,'UniformOutput',false),1:length(dgtda.Ui));
dgtda.time_embedding = cputime-dgtda.time_embedding;

train_proj=reshape(train_proj,[prod([ranks,I(length(ranks)+1:end)]),N]);
test_proj=reshape(test_proj,[prod([ranks,I(length(ranks)+1:end)]),tSize]);
MDAst = cputime;
[dgtda.PreLabel, dgtda.PreErr] = Classfier_KNN(train_proj, data.trLbl, test_proj, data.tsLbl, KNN);
dgtda.time_classify = cputime - MDAst;   % time to classify data
S1 = whos('train_proj');
Ui = dgtda.Ui;
dgtda.Ui=[];
S2 = whos('Ui');
dgtda.Storage  = (S1.bytes+S2.bytes)/numel(data.train)/8;

end