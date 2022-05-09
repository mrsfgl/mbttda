function [Us, outputs, Ys] = Man_alternatingmodes(Xs, classes, varargin)
% [Us, outsmode1, outsmode2, outerwhile, Ys] = ManPDA(Xs, classes, varargin)
% classes: vector containing class labels. Classes must be sequential
% numbers starting from one.
% varargin{1}: lowerdims
% varargin{2}: Us
% varargin{3}: opts
% varargin{4}: usestoppingcrit
% varargin{5}: maxits

%rng(0)
Xsample1 = Xs{1};
sizeX = size(Xsample1);

if length(sizeX) > 2
    error(['ManPDA.m: Input data has more than two dimensions. '...
        'This function is only customised for two-dimensional (i.e. matrix) data.'])
end

nmodes = length(sizeX);
nsamples = length(Xs);
Fdifftol = 1e-10;
Udifftol = 1e-12;
maxits = 1000;
Us = [];
optmeth = 'ManOpt';
lowerdims = sizeX;
inputfunc = @tensorsldaobj_matrixdata;

usestoppingcrit = true;
opts = [];

if ~isempty(varargin)
    if ~isempty(varargin{1})
        lowerdims = varargin{1};
    end
    
    if length(varargin)>=2 && ~isempty(varargin{2})
        Us = varargin{2};
    end
    
    if length(varargin)>=3 && ~isempty(varargin{3})
        opts = varargin{3};
    end
    
    if length(varargin)>=4 && ~isempty(varargin{4})
        usestoppingcrit = varargin{4};
    end
    
    if length(varargin)>=5 && ~isempty(varargin{5})
        maxits = varargin{5};
    end
    
    if length(varargin)>=6 && ~isempty(varargin{6})
        store = varargin{6};
    end
    
    if length(varargin)>=7 && ~isempty(varargin{7})
        if ismember(varargin{7}, {'ManOpt', 'bo13', 'wen12'})
            optmeth = varargin{7};
        else
            warning('ManPDA.m: chosen manifold optimiser is not an option. Using ManOpt.')
        end
    end
    
    if length(varargin)>=8 && ~isempty(varargin{8})
        inputfunc = varargin{8};
    end
    
    
end


if isempty(Us)
    Us = cell(1, nmodes);
    for kmode = 1:nmodes
        Us{kmode} = orth(randn(sizeX(kmode), lowerdims(kmode)));
    end
end


[N, K1] = size(Us{1});
[M, K2] = size(Us{2});

[cmean_m_xmeans, xi_m_cmeans, nis] = classbased_differences(Xs, classes);

U.U1 = Us{1};
U.U2 = Us{2};
% calculate Rw and Rb
[~, ~, Rw, Rb] = inputfunc(U,...
    cmean_m_xmeans, xi_m_cmeans, nis, K1, K2);

maxinnerits = 100;

opts.intialtau = -1;
opts.mxitr = maxinnerits;
opts.record = 1;
options.maxiter = maxinnerits;

switch optmeth
    case 'ManOpt'
        manifolds{1} = stiefelfactory(size(Us{1}, 1), size(Us{1}, 2));
        manifolds{2} = stiefelfactory(size(Us{2}, 1), size(Us{2}, 2));
        fvals = [];
        outs = cell(maxits, 2);
        for iit = 1:maxits
            oldU1 = U.U1;
            oldU2 = U.U2;
            for imode = 1:2
                problem.M = manifolds{imode};
                othermode = setdiff(1:2, imode);
                % Define the problem cost function and its Euclidean gradient.
                problem.cost  = @(U, store) mycost(Us{imode}, store,...
                    cmean_m_xmeans, xi_m_cmeans, nis,...
                    K1, K2, Rw, Rb, imode, Us{othermode}, inputfunc);
                
                problem.egrad = @(U, store) mygrad(Us{imode}, store,...
                    cmean_m_xmeans, xi_m_cmeans, nis,...
                    K1, K2, Rw, Rb, imode, Us{othermode}, inputfunc);
                
                
                % Solve.
                [Us{imode}, ~, outs{iit, imode}] = conjugategradient(problem, Us{imode}, options);
                fvals = [fvals cell2mat({outs{iit, imode}.cost})];
            end
            
            %if norm(U.U1-oldU1)<1e-10 && norm(U.U2-oldU2)<1e-10 
            %break
            %end
        end
        outs = outs(1:iit, :);
        
    case 'bo13'
        [U, outs] = OptStiManAFBB_myvariant(U,...
            @parafacldaobj_matrixdata, opts, cmean_m_xmeans, xi_m_cmeans,...
            nis, K1, K2, Rw, Rb);
        fvals = outs.FArray;
        
    case 'wen12'
        [U, outs] = OptStiefelGBB_myvariant(U,...
            @parafacldaobj_matrixdata, opts, cmean_m_xmeans, xi_m_cmeans,...
            nis, K1, K2, Rw, Rb);
        fvals = outs.fvals;
        
    otherwise
        error(['ManPDA.m: unknown optimisation method (' optmeth ').'])
end

outputs.fvals = fvals;
outputs.outs = outs;

Us{1} = U.U1;%U(1:N, 1:K1);
Us{2} = U.U2;%U((N+1):end, (K1+1):end);

if nargout >= 5
    Ys = cell(1, nsamples);
    for isample = 1:nsamples
        Yi = Xs{isample};
        for jmode = 1:nmodes % do multiplication between Xi tensor and U matrices
            origdims = size(Yi);
            Yi = Us{jmode}'*matricizing(Yi, jmode);
            newdims  = origdims;
            newdims(jmode) = size(Us{jmode}, 2);
            Yi = unmatricizing(Yi, jmode, newdims);
        end
        Ys{isample} = Yi;
    end
end
end

function [F, store] = mycost(x, store, classmeandiffs, observationdiffs,...
    nis, K1, K2, Rw, Rb, imode, x2, inputfunc)

if imode == 1
    U.U1 = x;
    U.U2 = x2;
else
    U.U1 = x2;
    U.U2 = x;
end
[F, ~, ~, ~, store]...
    = inputfunc(U,...
    classmeandiffs, observationdiffs, nis, K1, K2, ...
    Rw, Rb, store);
end


function [G, store] = mygrad(x, store, classmeandiffs, observationdiffs,...
    nis, K1, K2, Rw, Rb, imode, x2, inputfunc)

if imode == 1
    U.U1 = x;
    U.U2 = x2;
else
    U.U1 = x2;
    U.U2 = x;
end
[~, G, ~, ~, store]...
    = inputfunc(U,...
    classmeandiffs, observationdiffs, nis, K1, K2, ...
    Rw, Rb, store);

G = eval(['G.U' num2str(imode)]);
end

