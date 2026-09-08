function THWTTDA = thw_btt(data, parameters)
    % THW_BTT Three-branch block tensor-train discriminant analysis.
    % Samples are columns; parameters.I gives each sample's tensor shape.
    % See algorithm_parameters for shared options and docs/code-guide.md for layouts.
    % Author: Seyyid Emre Sofuoglu.
    % Uses the block TT eigensolver of Dolgov, Khoromskij, Oseledets, and
    % Savostyanov, Comput. Phys. Commun. (2014), doi:10.1016/j.cpc.2013.12.017.

    parameters = algorithm_parameters(parameters.I, parameters);
    data = validate_dataset(data, parameters.I);

    % Fold class-contiguous observations into the branch layout.
    k = computeK(parameters.I, 3);

    modeCount = length(parameters.I);     % tensor data order
    trainingCount = size(data.train, 2);      % train data size
    samplesPerClass = sum(data.trLbl == data.trLbl(1));
    classCount = trainingCount / samplesPerClass;
    leftDimension = prod(parameters.I(1:k(1)));
    middleDimension = prod(parameters.I(k(1) + 1:k(2)));
    rightDimension = prod(parameters.I(k(2) + 1:modeCount));
    branchModes{1} = 1:k(1);
    branchModes{2} = k(1) + 1:k(2);
    branchModes{3} = k(2) + 1:modeCount;

    data.train = reshape(data.train, [parameters.I, trainingCount]);
    data.test = reshape(data.test, [parameters.I, size(data.test, 2)]);
    data.train = permute(data.train, [branchModes{1}, modeCount + 1, branchModes{2}, branchModes{3}]);
    data.train = reshape(data.train, [leftDimension, samplesPerClass, classCount, middleDimension, rightDimension]);

    initializationTimer = tic;
    % Initialize branch factors
    cores = U2Ui_tau(reshape(data.train, [parameters.I(1:k(1)), trainingCount * rightDimension * middleDimension]), parameters.tau, parameters.rndI);
    cores = cores(1:k(1));
    leftRank = size(cores{k(1)}, 3);
    leftProjection = reshape(merge_tensor(cores(1:k(1))), [], leftRank);
    leftProjectedTraining = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 5, 1]);
    cores(k(1) + 1:k(2) + 1) = U2Ui_tau(reshape(leftProjectedTraining, [parameters.I(k(1) + 1:k(2)), trainingCount * rightDimension * leftRank]), parameters.tau, parameters.rndI);
    cores = cores(1:k(2));
    middleRank = size(cores{k(2)}, 3);
    middleProjection = reshape(merge_tensor(cores(k(1) + 1:k(2))), [], middleRank);
    middleProjectedTraining = permute(tmprod(leftProjectedTraining, middleProjection', 1), [4, 2, 3, 5, 1]);
    cores(k(2) + 1:modeCount + 1) = U2Ui_tau(reshape(middleProjectedTraining, [parameters.I(k(2) + 1:modeCount), trainingCount * leftRank * middleRank]), parameters.tau, parameters.rndI);
    cores = cores(1:modeCount);
    rightRank = size(cores{modeCount}, 3);
    rightProjection = reshape(merge_tensor(cores(k(2) + 1:modeCount)), [], rightRank);

    data.test = permute(reshape(data.test, leftDimension, middleDimension, rightDimension, []), [1, 4, 2, 3]);

    rightProjectedTraining = reshape(tmprod(data.train, {middleProjection', rightProjection'}, [4, 5]), [leftDimension, samplesPerClass, classCount, middleRank * rightRank]);
    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Left operations
        [cores(1:k(1)), l_sub_time] = lrnU_btt(rightProjectedTraining, cores(1:k(1)), parameters);
        leftProjection = reshape(merge_tensor(cores(1:k(1))), [], leftRank);
        % Middle Operations
        rmtt = tic;
        leftProjectedTraining = reshape(permute(tmprod(data.train, {leftProjection', rightProjection'}, [1, 5]), [4, 2, 3, 5, 1]), [middleDimension, samplesPerClass, classCount, rightRank * leftRank]);
        rmtt = toc(rmtt);
        [cores(k(1) + 1:k(2)), m_sub_time] = lrnU_btt(leftProjectedTraining, cores(k(1) + 1:k(2)), parameters);
        middleProjection = reshape(merge_tensor(cores(k(1) + 1:k(2))), [], middleRank);
        % Right Operations
        mmtt = tic;
        middleProjectedTraining = reshape(permute(tmprod(data.train, {leftProjection', middleProjection'}, [1, 4]), [5, 2, 3, 1, 4]), [rightDimension, samplesPerClass, classCount, leftRank * middleRank]);
        mmtt = toc(mmtt);
        [cores(k(2) + 1:modeCount), r_sub_time] = lrnU_btt(middleProjectedTraining, cores(k(2) + 1:modeCount), parameters);
        rightProjection = reshape(merge_tensor(cores(k(2) + 1:modeCount)), [], rightRank);

        learningSeconds = toc(learningTimer);
        % Extract training and test features.
        stageTimer = tic;
        lmtt = tic;
        rightProjectedTraining = reshape(tmprod(data.train, {middleProjection', rightProjection'}, [4, 5]), [leftDimension, samplesPerClass, classCount, middleRank * rightRank]);
        lmtt = toc(lmtt);
        projectedTraining = reshape(tmprod(rightProjectedTraining, leftProjection', 1), [leftRank, trainingCount, middleRank, rightRank]);
        trainingFeatures = ndim_unfold(projectedTraining, 2)';

        for i = 1:modeCount
            ablInd = randi(parameters.I(i), parameters.nA(i), 1);
            cores{i}(:, ablInd, :) = 0;
        end
        leftTestProjection = reshape(merge_tensor(cores(1:k(1))), [], leftRank);
        middleTestProjection = reshape(merge_tensor(cores(k(1) + 1:k(2))), [], middleRank);
        rightTestProjection = reshape(merge_tensor(cores(k(2) + 1:modeCount)), [], rightRank);
        testFeatures = project_branch_features(data.test, {leftTestProjection, middleTestProjection, rightTestProjection});
        THWTTDA.time_embedding(iter) = toc(stageTimer);   % time to embed data

        THWTTDA.time_subspace(iter) = learningSeconds + (iter == 1) * initializationSeconds;
        THWTTDA.time_mat(iter) = rmtt + mmtt + lmtt;
        THWTTDA.time_search(iter) = l_sub_time + m_sub_time + r_sub_time;

        stageTimer = tic;
        [THWTTDA.PreLabel{iter}, THWTTDA.PreErr(:, iter)] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);
        THWTTDA.time_classify(iter) = toc(stageTimer);   % time to classify data
    end
    THWTTDA.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    coreStorage = whos('cores');
    featureStorage = whos('trainingFeatures');

    THWTTDA.Ui = [];
    THWTTDA.Storage = (coreStorage.bytes + featureStorage.bytes) / (8 * numel(data.train));
end
