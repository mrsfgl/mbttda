function TTDA = twttda(data, parameters)
    % TWTTDA Two-branch Tensor-Train Discriminant Analysis (paper Algorithm 2).
    % Samples are columns; parameters.I gives each sample's tensor shape.
    % See algorithm_parameters for shared options and docs/code-guide.md for layouts.
    % Author: Seyyid Emre Sofuoglu.

    parameters = algorithm_parameters(parameters.I, parameters);
    data = validate_dataset(data, parameters.I);
    opts = optimizer_options();

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

    initializationTimer = tic;
    % Initialize branch factors
    cores = U2Ui_tau(reshape(data.train, [parameters.I(1:k), trainingCount * rightDimension]), parameters.tau, parameters.rndI);
    cores = cores(1:k);
    leftRank = size(cores{k}, 3);
    leftProjection = reshape(merge_tensor(cores(1:k)), [], leftRank);
    leftProjectedTraining = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 1]);
    cores(k + 1:modeCount + 1) = U2Ui_tau(reshape(leftProjectedTraining, [parameters.I(k + 1:modeCount), trainingCount * leftRank]), parameters.tau, parameters.rndI);
    cores = cores(1:modeCount);
    rightRank = size(cores{modeCount}, 3);
    rightProjection = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);

    data.test = permute(reshape(data.test, leftDimension, rightDimension, []), [1, 3, 2]);

    TTDA.objVal = [];

    rightProjectedTraining = tmprod(data.train, rightProjection', 4);
    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Left operations
        [cores(1:k), tmp] = lrnU(rightProjectedTraining, cores(1:k), parameters, opts);
        TTDA.objVal = [TTDA.objVal, tmp];
        leftProjection = reshape(merge_tensor(cores(1:k)), [], leftRank);
        % Right Operations
        leftProjectedTraining = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 1]);
        [cores(k + 1:modeCount), tmp] = lrnU(leftProjectedTraining, cores(k + 1:modeCount), parameters, opts);
        TTDA.objVal = [TTDA.objVal, tmp];
        rightProjection = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);

        learningSeconds = toc(learningTimer);
        % Extract training and test features.
        stageTimer = tic;
        rightProjectedTraining = tmprod(data.train, rightProjection', 4);
        projectedTraining = reshape(tmprod(rightProjectedTraining, leftProjection', 1), [leftRank, trainingCount, rightRank]);
        trainingFeatures = ndim_unfold(projectedTraining, 2)';
        for i = 1:modeCount
            ablInd = randi(parameters.I(i), parameters.nA(i), 1);
            cores{i}(:, ablInd, :) = 0;
        end
        leftTestProjection = reshape(merge_tensor(cores(1:k)), [], leftRank);
        rightTestProjection = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);
        testFeatures = project_branch_features(data.test, {leftTestProjection, rightTestProjection});

        TTDA.time_embedding(iter) = toc(stageTimer);   % time to embed data
        TTDA.time_subspace(iter) = learningSeconds + (iter == 1) * initializationSeconds;

        % Classify projected observations.

        stageTimer = tic;
        [TTDA.PreLabel{iter}, TTDA.PreErr(:, iter)] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);
        TTDA.time_classify(iter) = toc(stageTimer);   % time to classify data
    end
    TTDA.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    coreStorage = whos('cores');
    featureStorage = whos('trainingFeatures');
    TTDA.Ui = [];

    TTDA.Storage = (coreStorage.bytes + featureStorage.bytes) / (8 * numel(data.train));
end
