function TTDA = ttda(data, parameters)
    %  Tensor Train Discriminant Analysis.
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
    modeCount   = length(parameters.I);     % tensor data order
    trainingCount   = size(data.train, 2);      % train data size
    samplesPerClass = length(find(data.trLbl == data.trLbl(1)));
    classCount = trainingCount / samplesPerClass;

    data.train = reshape(data.train, [parameters.I, samplesPerClass, classCount]);
    data.test = reshape(data.test, [parameters.I, size(data.test, 2)]);
    [parameters.I, sI] = sort(parameters.I, 'descend');
    parameters.nA = parameters.nA(sI);
    data.train = permute(data.train, [sI, modeCount + 1, modeCount + 2]);
    data.train = reshape(data.train, [prod(parameters.I), samplesPerClass, classCount]);
    data.test = reshape(permute(data.test, [sI, modeCount + 1]), prod(parameters.I), []);
    initializationTimer = tic;
    % Initialize Ui
    cores = U2Ui_tau(reshape(data.train, [parameters.I, trainingCount]), parameters.tau, parameters.rndI); % U_1, ....,U_k, A
    cores = cores(1:end - 1);
    r = size(cores{end}, 3);
    TTDA.objVal = [];

    %% Loop
    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Learn Subspaces
        [cores, objVal, scatterTime, searchTime] = lrnU(data.train, cores, parameters, opts);
        TTDA.objVal = [TTDA.objVal, objVal];
        for i = 1:modeCount
            ablInd = randi(parameters.I(i), parameters.nA(i), 1);
            cores{i}(:, ablInd, :) = 0;
        end
        Sub  = reshape(merge_tensor(cores), [], r);
        TTDA.Ui(1:modeCount) = cores;

        learningSeconds = toc(learningTimer);
        % Projections
        TTLDA_start = tic;
        trainingFeatures = reshape(tmprod(data.train, Sub', 1), [r, trainingCount]);
        TTDA.Ui{modeCount + 1} = trainingFeatures;
        testFeatures = tmprod(data.test, Sub', 1);

        TTDA.time_embedding(iter) = toc(TTLDA_start);   % time to embed data
        TTDA.time_search(iter) = searchTime;
        TTDA.time_subspace(iter) = learningSeconds + (iter == 1) * initializationSeconds;

        % Classifications

        TTLDA_start = tic;
        [TTDA.PreLabel{iter}, TTDA.PreErr(:, iter)] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);
        TTDA.time_classify(iter) = toc(TTLDA_start);   % time to classify data
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
