function ThW = ThWTTDA(train_data, train_label, test_data,test_label, tensor_shape, tau, param, KNN)
% train_data  & test_data:  (I1*...*In) * N matrix
% train_label & test_label: 1 * N

numBranch = 3;
% define parameters
k   = computeK(tensor_shape, numBranch);

n   = length(tensor_shape);     % data order
N   = size(train_data, 2);      % data size = numClasses*numSamples
cSize = length(find(train_label==train_label(1))); % Number of samples
numC = N/cSize;                 % number of classes
Ls  = prod(tensor_shape(1:k(1)));
Ms  = prod(tensor_shape(k(1)+1:k(2)));
Rs  = prod(tensor_shape(k(2)+1:n));

% Initialize Us
Ui = inU(train_data, tensor_shape, numBranch, tau);


%% Loop
rprojTrain=reshape(train_data,[Ls,cSize,numC,Ms*Rs]);
for iter =1:param.maxiterL
    % Branch Operations 
    [Ui, sbS, ThW.time_subspace(iter)] = branchOp(train_data, Ui, mD, tensor_shape);
    
    % Projections
    TTLDA_start = cputime;
    lmtt = cputime;
    rprojTrain = reshape(tmprod(train_data,{MSub',RSub'},[4,5]),[Ls,cSize,numC,rS(2)*rS(3)]);
    lmtt = cputime - lmtt;
    ThW.Ui{iter,k(1)+1} = reshape(tmprod(rprojTrain,LSub',1),[rS(1),N,rS(2),rS(3)]);
    train_proj = ndim_unfold(ThW.Ui{iter,k(1)+1},2)';
    test_proj = ndim_unfold(tmprod(test_data,{LSub',MSub',RSub'},[1,3,4]),2)';
    ThW.time_embedding(iter) = cputime - TTLDA_start;   % time to embed data
    
    ThW.time_subspace(iter) = Lmattime + Mmattime + Rmattime + ...
        Lsearchtime + Msearchtime + Rsearchtime + lmtt + mmtt + rmtt;    % time to find subspace
    ThW.time_mat(iter) = Lmattime + Mmattime + Rmattime + rmtt + mmtt + lmtt;
    ThW.time_search(iter) = Lsearchtime + Msearchtime + Rsearchtime;
    
    TTLDA_start = cputime;
    [ThW.PreLabel{iter}, ThW.PreErr(:,iter)] = Classfier_KNN(train_proj, train_label, test_proj, test_label, KNN.K);
    ThW.time_classify(iter) = cputime - TTLDA_start;   % time to classify data
    ThW.Storage(iter)  = Dim_TT(Ui(1:n))+numel(ThW.Ui{iter,k(1)+1});
    %% End of Loop
end

end

function [Ui] = inU(Ytr, tensor_shape, numBranch, tau)

% Define parameters
if ~iscell(tensor_shape)
    for i = 1:length(tensor_shape)
        temp{i} = tensor_shape(i);
    end
    tensor_shape = temp; clear temp
end
k   = computeK(tensor_shape, numBranch);% Find the optimal separation of branches.
n   = length(tensor_shape);     % data order

mD{1} = 1:k(1);
d{1}  = prod(cell2mat(tensor_shape(mD{1})));
for i = 2:numBranch-1
    mD{i} = k(i-1)+1:k(i);  % Modes of each branch
    d{i}  = prod(cell2mat(tensor_shape(mD{i}))); % Dimension of each branch
end
mD{i+1} =  k(i)+1:n;
d{i+1} = prod(cell2mat(tensor_shape(mD{i+1})));

if iscell(Ytr)
    % Unfold the data if it is in the form of cells.
    nYtr = [];
    for i=1:length(Ytr)
        nYtr = cat(nYtr, Ytr{i}, 2);
    end
    Ytr = nYtr;
    clear nYtr
end
% Unfold the data matrix into a tensor and reshape.
Ytr = reshape(Ytr, tensor_shape{:}, []);
Ytr = permute(Ytr, [mD{:}, n+1]);
Ytr = reshape(Ytr, d{:}, []);

% Initialize Ui
Yn = Ytr;
for i = 1:numBranch
    Ui{i} = U2Ui_tau(reshape(Yn, tensor_shape{mD{i}}, []), tau); % Apply TT Decomposition to projected tensor.
    Ui{i} = Ui{i}(1:end-1);
    rS(i) = size(Ui{i}{end}, 3);
    sb{i} = reshape(merge_tensor(Ui{i}), [], rS(i));
    Yn    = permute(tmprod(Yn, sb{i}', 1), [2:numBranch+1, 1]);
end
end

function [Ui, sbS, comptime] = branchOp(Yi, Ui, mD, tensor_shape)
%  Updates all branches.
%

% Define parameters
nB  = length(Ui);
for i = 1:nB
    rS(i) = size(Ui{i}{end}, 3);
    d{i}  = prod(cell2mat(tensor_shape{mD{i}}));
    sbS   = reshape(merge_tensor(Ui{i}), [], rS(i));
end
n   = length(tensor_shape);             % data order

sbS = cellfun(@transpose, sbS, 'UniformOutput', false);
comptime = cputime;
if iscell(Yi)
    for branch = 1:nB
        ind = setdiff(1:nB, branch);
        for i=1:length(Yi)
            % Unfold the data matrix into a tensor and reshape.
            Yi{i} = reshape(Yi{i}, tensor_shape{:}, []);
            Yi{i} = permute(Yi{i}, [mD{:}, n+1]);
            Yi{i} = reshape(Yi{i}, d{:}, []);
            Yo{i} = reshape(permute(tmprod(Yi{i}, sbS(ind), ind), [branch, nB+1, ind]), d(branch), [], prod(rS(ind)));
        end
        [Ui{branch}, ~, ~] = lrnU(Yo, Ui{branch}, param, opts);
        sbS{branch} = reshape(merge_tensor(Ui{branch}), [], rS(branch));
    end
else
    % Unfold the data matrix into a tensor and reshape.
    Yi = reshape(Yi, tensor_shape{:}, []);
    Yi = permute(Yi, [mD{:}, n+1]);
    Yi = reshape(Yi, d{:}, []);
    for branch = 1:nB
        ind = setdiff(1:nB, branch);
        Yo  = reshape(permute(tmprod(Yi, sbS(ind), ind), [branch, nB+(1:2), ind]),[d(branch), cSize, numC, []]);
        [Ui{branch}, ~, ~] = lrnU(Yo, Ui{branch}, param, opts);
        sbS{branch}  = reshape(merge_tensor(Ui{branch}), [], rS(branch));
    end
end
comptime = cputime-comptime;
end