function T = merge_tensor(Ui, varargin)
    % Ui is a list of n 3-order tensor
    % T  is a tensor of order n+2 after merging Ui
    if length(varargin)==1
        sfirst=varargin{1};
    else
        sfirst=true;
    end
    n = length(Ui);
    for i=1:n
        if i==1
            T = Ui{1};
        else
            T = wwq_tensordot(T, Ui{i}, [i+1], [1]);
        end
    end
    if sfirst
        T = wwq_squeeze( T );
    end
end