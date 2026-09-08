function TTDA = ttda(data, parameters)
    % TTDA One-branch Tensor-Train Discriminant Analysis (paper Algorithm 1).
    % Samples are columns; parameters.I gives each sample's tensor shape.
    % See algorithm_parameters for shared options and docs/code-guide.md for layouts.
    % Author: Seyyid Emre Sofuoglu.

    parameters = algorithm_parameters(parameters.I, parameters);
    data = validate_dataset(data, parameters.I);
    opts = optimizer_options();

    % Fold class-contiguous observations into the branch layout.
    modeCount = length(parameters.I);     % tensor data order
    trainingCount = size(data.train, 2);      % train data size
    samplesPerClass = sum(data.trLbl == data.trLbl(1));
    classCount = trainingCount / samplesPerClass;

    data.train = reshape(data.train, [parameters.I, samplesPerClass, classCount]);
    data.test = reshape(data.test, [parameters.I, size(data.test, 2)]);
    [parameters.I, sI] = sort(parameters.I, 'descend');
    parameters.nA = parameters.nA(sI);
    data.train = permute(data.train, [sI, modeCount + 1, modeCount + 2]);
    data.train = reshape(data.train, [prod(parameters.I), samplesPerClass, classCount]);
    data.test = reshape(permute(data.test, [sI, modeCount + 1]), prod(parameters.I), []);
    initializationTimer = tic;
    % Initialize branch factors
    cores = U2Ui_tau(reshape(data.train, [parameters.I, trainingCount]), parameters.tau, parameters.rndI);
    cores = cores(1:end - 1);
    outputRank = size(cores{end}, 3);
    TTDA.objVal = [];

    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Learn Subspaces
        [cores, objVal, ~, searchTime] = lrnU(data.train, cores, parameters, opts);
        TTDA.objVal = [TTDA.objVal, objVal];
        for i = 1:modeCount
            ablInd = randi(parameters.I(i), parameters.nA(i), 1);
            cores{i}(:, ablInd, :) = 0;
        end
        projection = reshape(merge_tensor(cores), [], outputRank);

        learningSeconds = toc(learningTimer);
        % Extract training and test features.
        stageTimer = tic;
        trainingFeatures = reshape(tmprod(data.train, projection', 1), [outputRank, trainingCount]);
        testFeatures = tmprod(data.test, projection', 1);

        TTDA.time_embedding(iter) = toc(stageTimer);   % time to embed data
        TTDA.time_search(iter) = searchTime;
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
