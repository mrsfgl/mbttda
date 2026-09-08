function MPS = mps(data, parameters)
    % MPS Matrix product state features from sequential TT-PCA.
    % Samples are columns; parameters.I gives each sample's tensor shape.
    % See algorithm_parameters for shared options and docs/code-guide.md for layouts.
    % Author: Seyyid Emre Sofuoglu.

    parameters = algorithm_parameters(parameters.I, parameters);
    data = validate_dataset(data, parameters.I);

    % Fold class-contiguous observations into the branch layout.
    k = computeK(parameters.I, 2);
    modeCount = length(parameters.I);     % tensor data order
    trainingCount = size(data.train, 2);      % train data size
    samplesPerClass = sum(data.trLbl == data.trLbl(1));
    classCount = trainingCount / samplesPerClass;
    leftDimension = prod(parameters.I(1:k));
    rightDimension = prod(parameters.I(k + 1:modeCount));
    branchModes{1} = 1:k;
    branchModes{2} = k + 1:modeCount;

    data.train = reshape(data.train, [parameters.I, trainingCount]);
    data.test = reshape(data.test, [parameters.I, size(data.test, 2)]);
    data.train = permute(data.train, [branchModes{1}, modeCount + 1, branchModes{2}]);
    data.train = reshape(data.train, [leftDimension, samplesPerClass, classCount, rightDimension]);

    % Initialize branch factors
    stageTimer = tic;
    cores = U2Ui_tau(reshape(data.train, [parameters.I(1:k), trainingCount * rightDimension]), parameters.tau, parameters.rndI);
    cores = cores(1:k);
    leftRank = size(cores{k}, 3);
    leftProjection = reshape(merge_tensor(cores(1:k)), [], leftRank);
    leftProjectedTraining = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 1]);
    cores(k + 1:modeCount + 1) = U2Ui_tau(reshape(leftProjectedTraining, [parameters.I(k + 1:modeCount), trainingCount * leftRank]), parameters.tau, parameters.rndI);
    rightRank = size(cores{modeCount}, 3);
    rightProjection = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);
    MPS.time_subspace = toc(stageTimer);

    embeddingTimer = tic;

    data.test = permute(reshape(data.test, leftDimension, rightDimension, []), [1, 3, 2]);

    cores{modeCount + 1} = permute(reshape(cores{modeCount + 1}, rightRank, trainingCount, leftRank), [3, 2, 1]);
    trainingFeatures = ndim_unfold(cores{modeCount + 1}, 2)';
    testFeatures = project_branch_features(data.test, {leftProjection, rightProjection});

    MPS.time_embedding = toc(embeddingTimer);
    % Classify projected observations.
    classifyTimer = tic;
    [MPS.PreLabel, MPS.PreErr] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);

    MPS.time_classify = toc(classifyTimer);
    MPS.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    coreStorage = whos('cores');
    MPS.Storage = (coreStorage.bytes) / (8 * numel(data.train));
end
