function THWTTDA = thw_btt(data, parameters)
    %  Three-way Block Tensor Train Discriminant Analysis.
    %  -------------------------
    %  data: Struct that includes training and test data matrices and
    %  corresponding labels.
    %
    %  param : Struct of parameters.
    %    param.I: tensor shape of each sample:
    %    param.tau: Threshold parameter tau for TT-PCA.
    %    param.rndI: random initialization struct:
    %         param.rndI.flag: if true, initializes U's randomly. No TTPCA
    %         param.rndI.rank: needs to have elements for each tensor mode
    %         indicating the ranks of tensor factors if flag is true.
    %    param.maxIterOut: Outer loop max iteration.
    %    param.nA: Ablation indices for each mode. Sets that slice to zero.
    %    param.lambda: Parameter controlling the balance of within and
    %    between class scatters.
    %    param.display: Displays and extracts objective value.
    %    param.maxiter: Inner loop max iteration.
    %    param.error_tot: Inner loop error threshold.
    %  --------------------------
    % This function uses Tensor-Train Toolbox kindly
    % provided by [*]
    %
    %  Seyyid Emre Sofuoglu
    %
    % [*] Dolgov, Khoromskij, Oseledets, Savostyanov,
    % "Computation of extreme eigenvalues in higher dimensions using block
    % tensor train format", Comp. Phys. Comm. 2014, http://dx.doi.org/10.1016/j.cpc.2013.12.017
    %

    parameters = algorithm_parameters(parameters.I, parameters);
    data = validate_dataset(data, parameters.I);
    opts = optimizer_options();

    % Paper Sections III-IV: initialize, optimize branch factors, then classify.
    % read parameter
    k   = computeK(parameters.I, 3);

    modeCount   = length(parameters.I);     % tensor data order
    trainingCount   = size(data.train, 2);      % train data size
    samplesPerClass = length(find(data.trLbl == data.trLbl(1)));
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
    % Initialize Ui
    cores = U2Ui_tau(reshape(data.train, [parameters.I(1:k(1)), trainingCount * rightDimension * middleDimension]), parameters.tau, parameters.rndI); % U_1, ....,U_m
    cores = cores(1:k(1));
    leftRank = size(cores{k(1)}, 3);
    leftProjection = reshape(merge_tensor(cores(1:k(1))), [], leftRank);
    leftProjectedTraining = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 5, 1]);
    cores(k(1) + 1:k(2) + 1) = U2Ui_tau(reshape(leftProjectedTraining, [parameters.I(k(1) + 1:k(2)), trainingCount * rightDimension * leftRank]), parameters.tau, parameters.rndI); % U_{m+1}, ....,U_n
    cores = cores(1:k(2));
    middleRank = size(cores{k(2)}, 3);
    middleProjection = reshape(merge_tensor(cores(k(1) + 1:k(2))), [], middleRank);
    middleProjectedTraining = permute(tmprod(leftProjectedTraining, middleProjection', 1), [4, 2, 3, 5, 1]);
    cores(k(2) + 1:modeCount + 1) = U2Ui_tau(reshape(middleProjectedTraining, [parameters.I(k(2) + 1:modeCount), trainingCount * leftRank * middleRank]), parameters.tau, parameters.rndI); % U_{m+1}, ....,U_n
    cores = cores(1:modeCount);
    rightRank = size(cores{modeCount}, 3);
    rightProjection = reshape(merge_tensor(cores(k(2) + 1:modeCount)), [], rightRank);

    data.test = permute(data.test, [branchModes{1}, branchModes{2}, branchModes{3}, modeCount + 1]);
    data.test = permute(reshape(data.test, leftDimension, middleDimension, rightDimension, []), [1, 4, 2, 3]);

    %% Loop
    rightProjectedTraining = reshape(tmprod(data.train, {middleProjection', rightProjection'}, [4, 5]), [leftDimension, samplesPerClass, classCount, middleRank * rightRank]);
    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Left operations
        [cores(1:k(1)), l_sub_time] = lrnU_btt(rightProjectedTraining, cores(1:k(1)), parameters);
        leftProjection  = reshape(merge_tensor(cores(1:k(1))), [], leftRank);
        THWTTDA.Ui(1:k(1)) = cores(1:k(1));
        % Middle Operations
        rmtt = tic;
        leftProjectedTraining = reshape(permute(tmprod(data.train, {leftProjection', rightProjection'}, [1, 5]), [4, 2, 3, 5, 1]), [middleDimension, samplesPerClass, classCount, rightRank * leftRank]);
        rmtt = toc(rmtt);
        [cores(k(1) + 1:k(2)), m_sub_time] = lrnU_btt(leftProjectedTraining, cores(k(1) + 1:k(2)), parameters);
        middleProjection  = reshape(merge_tensor(cores(k(1) + 1:k(2))), [], middleRank);
        THWTTDA.Ui(k(1) + 2:k(2) + 1) = cores(k(2):-1:k(1) + 1);
        for i = k(1) + 2:k(2) + 1
            THWTTDA.Ui{i} = permute(THWTTDA.Ui{i}, [3, 2, 1]);
        end
        % Right Operations
        mmtt = tic;
        middleProjectedTraining = reshape(permute(tmprod(data.train, {leftProjection', middleProjection'}, [1, 4]), [5, 2, 3, 1, 4]), [rightDimension, samplesPerClass, classCount, leftRank * middleRank]);
        mmtt = toc(mmtt);
        [cores(k(2) + 1:modeCount), r_sub_time] = lrnU_btt(middleProjectedTraining, cores(k(2) + 1:modeCount), parameters);
        rightProjection  = reshape(merge_tensor(cores(k(2) + 1:modeCount)), [], rightRank);

        THWTTDA.Ui(k(2) + 2:modeCount + 1) = cores(modeCount:-1:k(2) + 1);
        for i = k(2) + 2:modeCount + 1
            THWTTDA.Ui{i} = permute(THWTTDA.Ui{i}, [3, 2, 1]);
        end

        learningSeconds = toc(learningTimer);
        % Projections
        TTLDA_start = tic;
        lmtt = tic;
        rightProjectedTraining = reshape(tmprod(data.train, {middleProjection', rightProjection'}, [4, 5]), [leftDimension, samplesPerClass, classCount, middleRank * rightRank]);
        lmtt = toc(lmtt);
        THWTTDA.Ui{k(1) + 1} = reshape(tmprod(rightProjectedTraining, leftProjection', 1), [leftRank, trainingCount, middleRank, rightRank]);
        trainingFeatures = ndim_unfold(THWTTDA.Ui{k(1) + 1}, 2)';

        for i = 1:modeCount
            ablInd = randi(parameters.I(i), parameters.nA(i), 1);
            cores{i}(:, ablInd, :) = 0;
        end
        leftTestProjection = reshape(merge_tensor(cores(1:k(1))), [], leftRank);
        middleTestProjection = reshape(merge_tensor(cores(k(1) + 1:k(2))), [], middleRank);
        rightTestProjection = reshape(merge_tensor(cores(k(2) + 1:modeCount)), [], rightRank);
        testFeatures = project_branch_features(data.test, {leftTestProjection, middleTestProjection, rightTestProjection});
        THWTTDA.time_embedding(iter) = toc(TTLDA_start);   % time to embed data

        THWTTDA.time_subspace(iter) = learningSeconds + (iter == 1) * initializationSeconds;
        THWTTDA.time_mat(iter) = rmtt + mmtt + lmtt;
        THWTTDA.time_search(iter) = l_sub_time + m_sub_time + r_sub_time;

        TTLDA_start = tic;
        [THWTTDA.PreLabel{iter}, THWTTDA.PreErr(:, iter)] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);
        THWTTDA.time_classify(iter) = toc(TTLDA_start);   % time to classify data
        %% End of Loop
    end
    THWTTDA.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    S1 = whos('cores');
    S2 = whos('trainingFeatures');
    cores = [];
    THWTTDA.Ui = [];
    tmp = data.train;
    S3 = whos('tmp');
    THWTTDA.Storage  = (S1.bytes + S2.bytes) / S3.bytes;
end
