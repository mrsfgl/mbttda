function [Us, objfuncval, Ys] = DGTDA(Xs, varargin)
% [Us, objfuncval, Ys] = DGTDA(Xs, classes, varargin)
% Mandatory input:
% Xs:           Cell array containing the observed tensors.
% classes:      Vector containing class labels. Classes must be sequential
%               numbers starting from one.
%
% Optional input:
% varargin{1}:  Vector containing the number of components to estimate for
%               each mode. Default: size of observations.
%
% Output:
% Us:           Cell array containing the projection matrices found.
% objfuncvals:  Value of the objective function that DGTDA tries to
%               optimise, with the final projection matrices in Us.
%
% li14:
%   Q. Li and D. Schonfeld,
%   'Multilinear Discriminant Analysis for Higher-Order Tensor
%   Data Classification',
%   IEEE Transactions on Pattern Analysis and Machine Intelligence

sizeX = size(Xs);
nmodes = ndims(Xs)-2;
sizeX = sizeX(1:nmodes);



if isempty(varargin) || isempty(varargin{1})
    lowerdims = sizeX;
else
    lowerdims = varargin{1};
    nmodes = length(lowerdims);
end

% calculate Xc - X for each class, where Xc is the class mean and X is the
% overall mean (stored in classmeandiffs) and Xcj - Xc where Xcj is the
% j'th observation from class c (stored in observationdiffs) and the number
% of observations from each class (stored in nis).
[classmeandiffs, observationdiffs] = classbased_differences(Xs);

Bs = cell(nmodes, 1);
for nmode = 1:nmodes
    diffmatricised = ndim_unfold(classmeandiffs, nmode);
    Bs{nmode} = diffmatricised*diffmatricised';
end


Ws = cell(nmodes, 1);
for nmode = 1:nmodes
    diffmatricised = ndim_unfold(observationdiffs, nmode);
    Ws{nmode} = diffmatricised*diffmatricised';
end

Us = cell(nmodes, 1);
for nmode = 1:nmodes
    if isempty(Ws{nmode})
        eta = 1;
    else
        eta = svds(Ws{nmode}\Bs{nmode},1);
    end
    [U, ~, ~] = svd(Bs{nmode} - eta*Ws{nmode},'econ');
    Us{nmode} = U(:, 1:lowerdims(nmode));
end

if nargout == 2
    Ys = tensor_projection(Xs, Us);
end

if nargout == 3
    if nmodes > 2
        error(['DGTDA.m: Input data has more than two dimensions. '...
            'This function is only customised for two-dimensional (i.e. matrix) data.'])
    end
    Ustruct.U1 = Us{1};
    Ustruct.U2 = Us{2};
    objfuncval = tensorsldaobj_matrixdata_normsratio(Ustruct, classmeandiffs, observationdiffs, nis, lowerdims(1), lowerdims(2));
end

end