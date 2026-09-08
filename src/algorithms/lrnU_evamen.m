function [U, seconds] = lrnU_evamen(Y, U, param)
    % LRNU_EVAMEN Compatibility name for the existing BTT eigensolver.
    [U, seconds] = lrnU_btt(Y, U, param);
end
