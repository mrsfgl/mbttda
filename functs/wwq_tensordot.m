function [T, o1, o2] = wwq_tensordot(T1, T2, O1, O2,varargin)
    %   merge T1 in order O1 with T2 in order O2
    if length(varargin)<1
        flag=true;
    else
        flag=varargin{1};
    end
    
    I1 = size(T1);
    I2 = size(T2);
    if length(I1)==2 && I1(2)~=1 && length(I2)~=2 && flag
        I1(3) =1;
    end
    if length(I2)==2 && I2(2)~=1 && length(I1)~=2 && flag
        I2(3) =1;
    end
    o1 = setdiff(1:length(I1), O1);
    o2 = setdiff(1:length(I2), O2);
    
    if isempty(o2)
        tp = [I1(o1), 1];
    elseif isempty(o1)
        tp = [1, I2(o2)];
    else
        tp = [I1(o1), I2(o2)];
    end
    
    T  = reshape(tens2mat(T1,o1,O1) * tens2mat(T2,O2,o2), tp);
end