function MDA = main_MDA(train_data, train_label, test_data,test_label, I, ranks, KNN)
% train_data  & test_data:  (I1*...*In) * N matrix
% train_label & test_label: 1 * N

% read parameter
n   = length(I);     % tensor data order
N   = size(train_data, 2);      % train data size
cSize = length(find(train_label==train_label(1)));
numC = N/cSize;


train_data=reshape(train_data,[I,cSize,numC]);
test_data=reshape(test_data,[I,size(test_data,2)]);

MDA.time_subspace = cputime;
MDA.Ui=DGTDA(train_data,ranks);
MDA.time_subspace = cputime-MDA.time_subspace;    % time to find subspace

% Projections
MDA.time_embedding = cputime;
train_proj = tmprod(train_data,cellfun(@transpose,MDA.Ui,'UniformOutput',false),1:length(I));
test_proj  = tmprod(test_data,cellfun(@transpose,MDA.Ui,'UniformOutput',false),1:length(I));
MDA.time_embedding = cputime-MDA.time_embedding;

train_proj=reshape(train_proj,[prod(ranks),N]);
test_proj=reshape(test_proj,[prod(ranks),size(test_proj,n+1)]);
MDAst = cputime;
[MDA.PreLabel, MDA.PreErr] = Classfier_KNN(train_proj, train_label, test_proj, test_label, KNN.K);
MDA.time_classify = cputime - MDAst;   % time to classify data
MDA.Storage  = sum(ranks.*I)+prod(ranks)*N;

end