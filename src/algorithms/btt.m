function BTT = btt(data, parameters)
    %  Block Tensor Train Discriminant Analysis.
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

    %% Pre-process
    modeCount = length(parameters.I);     % tensor data order
    num_tr_samp = size(data.train, 2);      % train data size
    class_size = length(find(data.trLbl == data.trLbl(1)));
    num_class = num_tr_samp / class_size;

    data.train = reshape(data.train, [prod(parameters.I), class_size, num_class]);
    initializationTimer = tic;
    %% Initialize Ui
    cores = U2Ui_tau(reshape(data.train, [parameters.I, num_tr_samp]), parameters.tau); % U_1, ....,U_k, A
    cores = cores(1:end - 1);
    r = size(cores{end}, 3);

    %% Loop
    initializationSeconds = toc(initializationTimer);
    for iter = 1:parameters.maxiterOut
        learningTimer = tic;
        % Learn Subspaces
        [cores, train_time] = lrnU_btt(data.train, cores, parameters);
        Sub = reshape(merge_tensor(cores), [], r);
        for i = 1:modeCount
            BTT.Ui{i} = full(cores{i});
        end

        learningSeconds = toc(learningTimer);
        % Projections
        btt_start = tic;
        trainingFeatures = reshape(tmprod(data.train, Sub', 1), [r, num_tr_samp]);
        BTT.Ui{modeCount + 1} = trainingFeatures;
        testFeatures = Sub' * data.test;

        BTT.time_embedding(iter) = toc(btt_start);   % time to embed data
        BTT.time_search(iter) = train_time;
        BTT.time_subspace(iter) = learningSeconds + (iter == 1) * initializationSeconds;

        % Classifications
        btt_start = tic;
        [BTT.PreLabel{iter}, BTT.PreErr(:, iter)] = Classfier_KNN(trainingFeatures, data.trLbl, testFeatures, data.tsLbl, 1);
        BTT.time_classify(iter) = toc(btt_start);   % time to classify data
        %% End of Loop
    end
    BTT.metrics = result_metrics(cores(1:modeCount), trainingFeatures, numel(data.train));
    S1 = whos('cores');
    S2 = whos('trainingFeatures');
    cores = [];
    BTT.Ui = [];
    tmp = data.train;
    S3 = whos('tmp');
    BTT.Storage = (S1.bytes + S2.bytes) / S3.bytes;

end
