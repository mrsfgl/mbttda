function [ U , searchtime] = lrnU_evamen(Y, U, param)
% [U, searchtime] = lrnU_evamen(Y, U, param)
% Returns tensor factors that minimize the within-class scatter and maximize
% between-class scatter with an ALS scheme.
% 
% [*] Kressner, Daniel, Michael Steinlechner, and André Uschmajew.
% "Low-rank tensor methods with subspace correction for symmetric 
%   eigenvalue problems." SIAM Journal on Scientific Computing 36.5 
%   (2014): A2346-A2368.
%
%  Seyyid Emre Sofuoglu
n = ndims(Y);
k = length(U);
r = size(U{k},3);
I = zeros(1,k);
rS = I;
for i=1:k
    I(i) = size(U{i},2);
    rS(i) = size(U{i},1);
end
num_class = size(Y,n-1);
class_size = size(Y,n-2);
if n==3
    num_class  = size(Y,n);
    class_size = size(Y,n-1);
end

% Construct Scatter Matrix
tic;
[Sw,Sb]=const_scat(Y,prod(I),class_size,num_class);
if param.lambda>0
    Z = Sw-param.lambda*Sb;
else
    param.lambda=svds(Sw\Sb,1);
    Z = Sw-param.lambda*Sb;
end
clear Sw Sb

Z = TTeMPS_op(core2cell(tt_matrix(reshape(Z, [I, I]), sqrt(param.tau))));
opts.maxiter = 3;
opts.maxrank = 8;
opts.tol = param.tau;
opts.precInner = false;
opts.verbose = 0;
U = amen_eigenvalue(Z, [], r, [rS,1], opts);
searchtime  = toc;
end

