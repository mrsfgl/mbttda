function [k] = computeK(shape, n)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
if iscell(shape)
    shape = cell2mat(shape);
end
l = length(shape);

if l<n
    error('The number of branches should not exceed the number of modes.');
elseif l == n
    k      = 1:n-1;
else
    s      = sort(nchoosek(1:(l-1),(n-1)),2);
    w      = prod(shape)^(1/n);
    for i=1:size(s,1)
        cs(i, 1) = prod(shape(1:s(i,1)));
        for j=2:(n-1)
            cs(i, j) = prod(shape(s(i,j-1)+1:s(i,j)));
        end
        cs(i, n) = prod(shape(s(i,n-1)+1:end));
    end
    [~, i] = min(sum(abs(cs-w),2));
    k      = s(i,:);
end

end

%%
% setSizes=findnumSets(l,n);
% 
% for i=1:size(setSizes,1)
%     [dists,modes]=dimComp(shape,setSizes(i,:));
%     [dMin(i),indMin(i)]=min(dists(:));
%     if n~=2
%         mMin{i}=modes(1:size(modes,1),indMin(i));
%     else
%         mMin{i}={modes(indMin(i),:)};
%     end
% end
% [~,i]=min(dMin);
% mMin=mMin{i};
% set=1:l;
% for i=1:n-1
%     set=setdiff(set,mMin{i});
% end
% mMin{n}=set;
% for i=1:n
%     [dim{i},ind]=sort(shape(mMin{i}),'descend');
%     mMin{i}=mMin{i}(ind);
% end