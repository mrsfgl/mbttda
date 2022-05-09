function auc = direct_predict(fun, x_train, x_test, ...
    y_train, y_test, ncomps)
if isequal(fun, @bilinear_logreg) || isequal(fun, @bilinear_logreg_tucker)
else
    error(['direct_predict.m: function handle given as ', ...
        'input must be one of @bilinear_logreg, or @bilinear_logreg_tucker.'])
end

nits=100;

ntrials = length(x_train);
[n_rows, n_cols] = size(x_train{1});
Xsmat_train = reshape(...
    cell2mat(x_train), [n_rows, n_cols, ntrials]);

ntrials = length(x_test);
Xsmat_test = reshape(...
    cell2mat(x_test), n_rows, n_cols, ntrials);

nrandinits = 5;
results = cell(nrandinits, 1);
objfuncvals = NaN(1, nrandinits);
opts.maxeval = nits;
for iit = 1:nrandinits
    [I, J, ~] = size(Xsmat_train);
    if isequal(fun, @bilinear_logreg_tucker)
        nparams = 1+I*ncomps+J*ncomps+ncomps^2-ncomps;
    else
        nparams = 1+I*ncomps+J*ncomps;
    end
    x0 = randn(nparams, 1)*1e-5;
    tic
    [X, info, perf] = ucminf(fun, x0, opts, [], Xsmat_train, y_train-1, ncomps);
    t = toc;
    
    finalxvector = X(:, end);
    
    if isequal(fun, @bilinear_logreg_tucker)
        w0 = finalxvector(1);
        Us{1} = reshape(finalxvector(2:(2+I*ncomps-1)),I,ncomps); % U in bilinear logreg code
        Us{2} = reshape(finalxvector((2+I*ncomps):(2+I*ncomps+J*ncomps-1)),...
            J,ncomps); % V in bilinear logreg code
        w = finalxvector(2+I*ncomps+J*ncomps:end);
        results{iit}.w = w;
    else
        w0 = finalxvector(1);
        Us{1} = reshape(finalxvector(2:(2+I*ncomps-1)),I,ncomps); % U in bilinear logreg code
        Us{2} = reshape(finalxvector((2+I*ncomps):end),J,ncomps); % V in bilinear logreg code
    end
    
    results{iit}.w0 = w0;
    results{iit}.Us = Us;
    results{iit}.X = X;
    results{iit}.info = info;
    results{iit}.perf = perf;
    results{iit}.t = t;
    if ~isempty(perf)
        results{iit}.objfuncvals = perf.f;
    end
    objfuncvals(iit) = info(1);
end

[~, bestit] = max(objfuncvals);



Us = results{bestit}.Us;
w0 = results{bestit}.w0;
XsU = tmult(Xsmat_test, Us{1}', 1);


if isequal(fun, @bilinear_logreg_tucker)
    W=nan(ncomps);
    ignore=logical(eye(ncomps));
    W(~ignore) = results{bestit}.w;
    W(ignore)=1;
    psiXs = squeeze(sum(sum(bsxfun(@times, tmult(XsU, Us{2}', 2), W),1),2));
    
else
    psiXs = squeeze(sum(sum(bsxfun(@times, XsU, Us{2}'),1),2)); % psiXs squeezed
end


linfunc = psiXs + w0; % vector of length ntrials
predprobsclassone = (1./(1+exp(-linfunc)));
predictions = [predprobsclassone 1-predprobsclassone];

[~, ~, ~, auc] = perfcurve(y_test, predictions(:, 2), 2);

%%%%%%%%









end