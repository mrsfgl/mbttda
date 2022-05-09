function [cmean_m_xmeans, xi_m_cmeans] = classbased_differences(Xs)
% [cmean_m_xmean, xi_m_cmean, nis] = classbased_differences(Xs)
%
% input:
% Xs: cell array of (multi-dimensional) matrices
%
% output:
% cmean_m_xmean: class means minus overall mean
% xi_m_cmean: observations minus corresponding class mean

nsamples = size(Xs,ndims(Xs)-1);
nclasses = size(Xs,ndims(Xs));

Xmean = mean(mean(Xs,ndims(Xs)),ndims(Xs)-1);
cm=mean(Xs,ndims(Xs)-1);

vec=ones(1,ndims(Xs));
vec(ndims(Xs)-1)=nsamples;
xi_m_cmeans = Xs-repmat(cm,vec);

vec=ones(1,ndims(Xs));
vec(ndims(Xs))=nclasses;
cmean_m_xmeans = cm-repmat(Xmean,vec);

end
