function LDA = myLDA(data, param)
    % data.train  & data.test:  (I1*...*In) * N matrix
    % data.trLbl & data.tsLbl: 1 * N
    
    % read parameter
    N   = size(data.train, 2);      % train data size
    cSize = size(data.train, 2);
    numC = size(data.train, 3);
    
    % Construct Scatter Matrix Z
    tic;
    Sw=data.train*(eye(size(data.train,2))-(kron(eye(numC),ones(cSize)/cSize)));
    Sw=Sw*Sw';
    Sb=data.train*(kron(eye(N/cSize),ones(cSize,1)/cSize)-ones(N,numC)/N);
    Sb=Sb*Sb';
    % Construct Z 
    Z = Sw-param.lambda*Sb;
    
    [Vrn, E] = eig(Z);                    % V is eigenvectors
    dE      = diag(E).^-1; r=sum(dE>param.tau*max(dE));
    Vrn = Vrn(:,end-r+1:end);
    tV      = toc;
    LDA.time_subspace    = tV;    % time to find subspace
    
    LDA_start = cputime;
    lda_trproj         = Vrn'*data.train;
    lda_testproj       = Vrn'*data.test;
    LDA.time_embedding = cputime - LDA_start;   % time to embed data
    
    
    ldastart=cputime;
    [LDA.PreLabel, LDA.PreErr] = Classfier_KNN(lda_trproj, data.trLbl, lda_testproj, data.tsLbl, 1);
    LDA.time_classify = cputime- ldastart;   % time to classify data
    S1 = whos('lda_trproj');
    S2 = whos('Vrn');
    A = data.train;
    S3 = whos('A');
    LDA.Storage = (S1.bytes+S2.bytes)/S3.bytes;
end