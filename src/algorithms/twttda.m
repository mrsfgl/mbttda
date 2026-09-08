function TTDA = twttda(data, parameters)
    %  2-Way Tensor Train Discriminant Analysis.
    %  Paper Algorithm 2: alternate left and right projected branch problems.
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

    initializationTimer = tic;
    % Initialize Ui
    cores = U2Ui_tau(reshape(data.train, [parameters.I(1:k), trainingCount * rightDimension]), parameters.tau, parameters.rndI); % U_1, ....,U_k, A
    cores = cores(1:k);
    leftRank = size(cores{k}, 3);
    leftProjection = reshape(merge_tensor(cores(1:k)), [], leftRank);
    leftProjectedTraining = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 1]);
    cores(k + 1:modeCount + 1) = U2Ui_tau(reshape(leftProjectedTraining, [parameters.I(k + 1:modeCount), trainingCount * leftRank]), parameters.tau, parameters.rndI); % U_1, ....,U_k, A
    cores = cores(1:modeCount);
    rightRank = size(cores{modeCount}, 3);
    rightProjection = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);

    data.test = permute(data.test, [branchModes{1}, branchModes{2}, modeCount + 1]);
    data.test = permute(reshape(data.test, leftDimension, rightDimension, []), [1, 3, 2]);

    TTDA.objVal = [];

    %% Loop
    rightProjectedTraining = tmprod(data.train, rightProjection', 4);
    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Left operations
        [cores(1:k), tmp, Lscattertime, Lsearchtime] = lrnU(rightProjectedTraining, cores(1:k), parameters, opts);
        TTDA.objVal = [TTDA.objVal, tmp];
        leftProjection         = reshape(merge_tensor(cores(1:k)), [], leftRank);
        TTDA.Ui(1:k) = cores(1:k);
        % Right Operations
        leftProjectedTraining = permute(tmprod(data.train, leftProjection', 1), [4, 2, 3, 1]);
        [cores(k + 1:modeCount), tmp, Rscattertime, Rsearchtime] = lrnU(leftProjectedTraining, cores(k + 1:modeCount), parameters, opts);
        TTDA.objVal = [TTDA.objVal, tmp];
        rightProjection         = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);

        TTDA.Ui(k + 2:modeCount + 1) = cores(modeCount:-1:k + 1);
        for i = k + 2:modeCount + 1
            TTDA.Ui{i} = permute(TTDA.Ui{i}, [3, 2, 1]);
        end

        learningSeconds = toc(learningTimer);
        % Projections
        stageTimer = tic;
        rightProjectedTraining = tmprod(data.train, rightProjection', 4);
        TTDA.Ui{k + 1} = reshape(tmprod(rightProjectedTraining, leftProjection', 1), [leftRank, trainingCount, rightRank]);
        trainingFeatures = ndim_unfold(TTDA.Ui{k + 1}, 2)';
        for i = 1:modeCount
            ablInd = randi(parameters.I(i), parameters.nA(i), 1);
            cores{i}(:, ablInd, :) = 0;
        end
        leftTestProjection = reshape(merge_tensor(cores(1:k)), [], leftRank);
        rightTestProjection = reshape(merge_tensor(cores(k + 1:modeCount)), [], rightRank);
        testFeatures = project_branch_features(data.test, {leftTestProjection, rightTestProjection});

        TTDA.time_embedding(iter) = toc(stageTimer);   % time to embed data
        TTDA.time_subspace(iter) = learningSeconds + (iter == 1) * initializationSeconds;

        % Classifications

        stageTimer = tic;
        [TTDA.PreLabel{iter}, TTDA.PreErr(:, iter)] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);
        TTDA.time_classify(iter) = toc(stageTimer);   % time to classify datUi

        %% End of Loop
    end
    TTDA.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    S1 = whos('cores');
    S2 = whos('trainingFeatures');
    TTDA.Ui = [];
    cores = [];
    tmp = data.train;
    S3 = whos('tmp');
    TTDA.Storage  = (S1.bytes + S2.bytes) / S3.bytes;

end
