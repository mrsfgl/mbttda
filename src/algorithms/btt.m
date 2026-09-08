function BTT = btt(data, parameters)
    % BTT One-branch block tensor-train discriminant analysis.
    % Samples are columns; parameters.I gives each sample's tensor shape.
    % See algorithm_parameters for shared options and docs/code-guide.md for layouts.
    % Author: Seyyid Emre Sofuoglu.
    % Uses the block TT eigensolver of Dolgov, Khoromskij, Oseledets, and
    % Savostyanov, Comput. Phys. Commun. (2014), doi:10.1016/j.cpc.2013.12.017.

    parameters = algorithm_parameters(parameters.I, parameters);
    data = validate_dataset(data, parameters.I);

    %% Pre-process
    modeCount = length(parameters.I);     % tensor data order
    trainingCount = size(data.train, 2);      % train data size
    samplesPerClass = sum(data.trLbl == data.trLbl(1));
    classCount = trainingCount / samplesPerClass;

    data.train = reshape(data.train, [prod(parameters.I), samplesPerClass, classCount]);
    initializationTimer = tic;
    %% Initialize branch factors
    cores = U2Ui_tau(reshape(data.train, [parameters.I, trainingCount]), parameters.tau);
    cores = cores(1:end - 1);
    outputRank = size(cores{end}, 3);

    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Learn Subspaces
        [cores, searchTime] = lrnU_btt(data.train, cores, parameters);
        projection = reshape(merge_tensor(cores), [], outputRank);

        learningSeconds = toc(learningTimer);
        % Extract training and test features.
        stageTimer = tic;
        trainingFeatures = reshape(tmprod(data.train, projection', 1), [outputRank, trainingCount]);
        testFeatures = projection' * data.test;

        BTT.time_embedding(iter) = toc(stageTimer);   % time to embed data
        BTT.time_search(iter) = searchTime;
        BTT.time_subspace(iter) = learningSeconds + (iter == 1) * initializationSeconds;

        % Classify projected observations.
        stageTimer = tic;
        [BTT.PreLabel{iter}, BTT.PreErr(:, iter)] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);
        BTT.time_classify(iter) = toc(stageTimer);   % time to classify data
    end
    BTT.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    coreStorage = whos('cores');
    featureStorage = whos('trainingFeatures');

    BTT.Ui = [];
    BTT.Storage = (coreStorage.bytes + featureStorage.bytes) / (8 * numel(data.train));
end
