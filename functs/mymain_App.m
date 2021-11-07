function App = mymain_App(train_data, train_label, ...
    test_data,test_label, tensor_shape, tau, para_App,KNN)
    % train_data  & test_data:  (I1*...*In) * N matrix
    % train_label & test_label: 1 * N
    
    % unitary solver parameter
    opts.record = 0;        
    opts.mxitr  = 1000;
    opts.gtol   = 1e-5;
    opts.xtol   = 1e-5;
    opts.ftol   = 1e-8;
    opts.tau    = 1e-3;
    
    % read parameter
    n   = length(tensor_shape);     % tensor data order
    N   = size(train_data, 2);      % train data size
    cSize = length(find(train_label==train_label(1)));
    numC = N/cSize;
    
    % Initialize Ui
%     Ui = U2Ui_tau(reshape(train_data, [tensor_shape, N]), tau); % U_1, ....,U_k, A 
%     Ui = Ui(1:n);                                               % XU_{k+1},..., U_n
    % Construct Scatter Matrix Z
    Sw=train_data*(eye(size(train_data,2))-(kron(eye(numC),ones(cSize)/cSize)));
    Sw=Sw*Sw';
    Sb=train_data*(kron(eye(N/cSize),ones(cSize,1)/cSize)-ones(N,numC)/N);
    Sb=Sb*Sb';
    % Construct Y, and thus Z 
%     Z = reshape(Sw-para_App.lambda*Sb,[tensor_shape,tensor_shape]);
    Z = Sw-para_App.lambda*Sb;
%     [App.Ui,App.Vrn, ~,tApp,tV] = Apro_Solver(Ui, Z, para_App, opts); 
    tV=cputime;
%     if rn<0.05*size(Z,1)
        [App.Vrn, E] = eig(Z);                    % V is eigenvectors
        dE=diag(E); r=sum(dE>tau*max(dE));
        App.Vrn=App.Vrn(:,1:r);
%     else
%         [V, ~] = eig(tens2mat(Z, 1:n, n+1:2*n));                    % V is eigenvectors
%         Vrn = V(:, 1: rn);                                          % rn are rn smallest eigen vectors
%     end
    tV=cputime-tV;
%     disp(['The objval is ', num2str(ObjVal(Z, App.Ui, n)), ' by Approximation']);
%     Subspace_App         = tens2mat(merge_tensor(App.Ui),1:n, n+1);
%     App.time_subspace    = tApp+tV;    % time to find subspace
    App.LDAtime_subspace    = tV;    % time to find subspace
    
%     TTNPE_start = cputime;
%     train_proj          = Subspace_App' * train_data;
%     test_proj           = Subspace_App' * test_data;
%     App.time_embedding = cputime- TTNPE_start;   % time to embed data
    
    LDA_start = cputime;
    lda_trproj          = App.Vrn'*train_data;
    lda_testproj          = App.Vrn'*test_data;
    App.LDAtime_embedding = cputime - LDA_start;   % time to embed data
    
%     TTNPE_start = cputime;
%     [App.PreLabel, App.PreErr] = Classfier_KNN(train_proj, train_label, test_proj, test_label, KNN.K);
%     App.time_classify = cputime- TTNPE_start;   % time to classify data
%     App.Storage  = Dim_TT(App.Ui) + size(App.Ui{end},3) * size(train_data, 2);
    
    ldastart=cputime;
    [App.LDAPreLabel, App.LDAPreErr] = Classfier_KNN(lda_trproj, train_label, lda_testproj, test_label, KNN.K);
    App.LDAtime_classify = cputime- ldastart;   % time to classify data
    App.LDAStorage = numel(App.Vrn) + size(App.Vrn,2)* size(train_data, 2);
end