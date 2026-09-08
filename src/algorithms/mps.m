function MPS = mps(data, parameters)
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

    parameters = algorithm_parameters(parameters.I, parameters);
    data = validate_dataset(data, parameters.I);
    opts = optimizer_options();

    % Paper Sections III-IV: initialize, optimize branch factors, then classify.
    % read parameter
    k   = computeK(parameters.I, 2);
    modeCount   = length(parameters.I);     % tensor data order
    trainingCount   = size(data.train, 2);      % train data size
    samplesPerClass = length(find(data.trLbl == data.trLbl(1)));
    classCount = trainingCount / samplesPerClass;
    leftDimension = prod(parameters.I(1:k));
    rightDimension = prod(parameters.I(k + 1:modeCount));
    branchModes{1} = 1:k;
    branchModes{2} = k + 1:modeCount;

    data.train = reshape(data.train, [parameters.I, trainingCount]);
    data.test = reshape(data.test, [parameters.I, size(data.test, 2)]);
    data.train = permute(data.train, [branchModes{1}, modeCount + 1, branchModes{2}]);
    data.train = reshape(data.train, [leftDimension, samplesPerClass, classCount, rightDimension]);

    % Initialize Ui
    stageTimer = tic;
    cores = U2Ui_tau(reshape(data.train, [parameters.I(1:k), trainingCount * rightDimension]), parameters.tau, parameters.rndI); % U_1, ....,U_k, A
    cores = cores(1:k);
    leftRank = size(cores{k}, 3);
    leftProjection         = reshape(merge_tensor(cores(1:k)), [], leftRank);
    leftProjectedTraining   = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 1]);
    cores(k + 1:modeCount + 1)  = U2Ui_tau(reshape(leftProjectedTraining, [parameters.I(k + 1:modeCount), trainingCount * leftRank]), parameters.tau, parameters.rndI); % U_1, ....,U_k, A
    rightRank = size(cores{modeCount}, 3);
    rightProjection         = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);
    MPS.time_subspace = toc(stageTimer);

    embeddingTimer = tic;
    data.test = permute(data.test, [branchModes{1}, branchModes{2}, modeCount + 1]);
    data.test = permute(reshape(data.test, leftDimension, rightDimension, []), [1, 3, 2]);

    cores{modeCount + 1} = permute(reshape(cores{modeCount + 1}, rightRank, trainingCount, leftRank), [3, 2, 1]);
    trainingFeatures = ndim_unfold(cores{modeCount + 1}, 2)';
    testFeatures = project_branch_features(data.test, {leftProjection, rightProjection});

    MPS.time_embedding = toc(embeddingTimer);
    % Classifications
    classifyTimer = tic;
    [MPS.PreLabel, MPS.PreErr] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);

    MPS.time_classify = toc(classifyTimer);
    MPS.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    S1 = whos('cores');
    tmp = data.train;
    S3 = whos('tmp');
    MPS.Storage  = (S1.bytes) / S3.bytes;

end
