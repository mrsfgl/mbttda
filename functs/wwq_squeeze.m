function T_out = wwq_squeeze(T_in)
    shape = size(T_in);
    if length(shape)<=2 && sum(shape==1)>=1
        T_out = T_in(:);
    else
        T_out = reshape(T_in, shape( shape ~= 1) );
    end
end