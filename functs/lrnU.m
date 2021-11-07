function [ U, objVal, Mattime, searchtime] = lrnU( data, U, param, varargin)
% Learn tensor factors U which minimize the discriminability. 
% Computes the scatter and minimizes a quadratic penalty. 

% initialize the parameters of the line search algorithm.
if ~isempty(varargin)
    opts = varargin{1};
else
    opts.record = 0;
    opts.mxitr  = 1000;
    opts.gtol   = 1e-5;
    opts.xtol   = 1e-5;
    opts.ftol   = 1e-8;
    opts.tau    = 1e-3;
end
% Initialize parameters
n       = ndims(data); % order of data, is folded.
k       = length(U);   % order of tensor, to unfold the data.
I       = zeros(1, k);  % size tuple intialization.
for i=1:k
    I(i) = size(U{i}, 2);
end


Mattime = cputime;
% Construct Scatter Matrix
searchtime = cputime;
[Sw,Sb] = const_scat(data);
if param.lambda>0
    Z = Sw-param.lambda*Sb;
else
    param.lambda=svds(Sw\Sb,1);
    Z = Sw-param.lambda*Sb;
end

Z = reshape(Z, [I, I]);
clear Sw Sb
[U, objVal] = TensNet_Solver(U, Z, param, opts);
searchtime=cputime-searchtime;
Mattime=cputime-Mattime;

end

