function [ Sw, Sb ] = const_scat(data)
% Construct Scatter Matrices: Computes between class (Sb) and within class
% (Sw) scatters. 
% 
n = ndims(data);
if n>3
    numC = size(data, n-1); % Number of Classes
elseif n==3 
    % If data is not projected, last mode is the class mode.
    numC = size(data, n);
elseif n==1
    % If data is unbalanced, it will be stored in a cell array.
    numC = length(data);
else 
    error('Number of modes of the data is weird!')
end
    
if n>=3
    Sw = zeros(size(data,1));
else
    Sw = zeros(size(data{1}, 1));
end
for i = 1:numC
    if n>3
        for j = 1:size(data,n)
            Sw = Sw + cov(data(:,:, i, j)');
        end
    elseif n==3
        Sw = Sw + cov(data(:,:,i)');
    else
        for j = 1:size(data{i}, ndims(data{i}))
            Sw = Sw + cov(data{i}(:,:, j)');
        end
    end
end
% LSw=data-repmat(mean(data,2),1,cSize,1,1);
% LSw=ndim_unfold(LSw,1);
% Sw=LSw*LSw';
% for i=1:N
%     Sw=Sw+sqz(LSw(:,:,i))*sqz(LSw(:,:,i))';
% end
if n>3
    Sb = zeros(size(Sw));
    for j = 1:size(data, n)
        Sb = Sb + cov(squeeze(mean(data(:,:,:,j), 2))');
    end
elseif n==3
    Sb = cov(squeeze(mean(data, 2))');
else
    M = [];
    for i = 1:numC
        M = cat(M, mean(data{i}(:,:,:) , 2), 3);
    end
    for j=1:size(M, 3)
        Sb = cov(M(:,:,j)');
    end
end
% LSb=sqz(mean(data, 2)-repmat(mean(mean(data,2),3),1,1,numC,1));
% LSb=ndim_unfold(permute(LSb,[1,3,2]),1);
% Sb=LSb*LSb';
end

