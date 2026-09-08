function values = method_tau_grid(method, config, preset)
    % METHOD_TAU_GRID Keep each method's original grid independent of its neighbors.
    key = upper(method);
    if ismember(key, {'CMDA', 'DGTDA'}) && ~isempty(config.ranks)
        values = 0; % Threshold is unused when the caller supplies explicit ranks.
        return
    end
    if isfield(config.tau_by_method, key)
        values = config.tau_by_method.(key);
    elseif ~isempty(config.tau)
        values = config.tau;
    else
        switch key
            case {'TTDA', 'BTT', 'TTNPE'}
                field = 'tau_list2';
            case {'TWTTDA', 'MPS', 'TWBTT', 'THWBTT'}
                field = 'tau_list3';
            case 'THW'
                field = 'tau_list4';
            otherwise
                field = 'tau_list';
        end
        if strcmp(config.preset_family, 'mda')
            field = 'tau_list';
        end
        if ~isfield(preset, field) || isempty(preset.(field))
            error('mbttda:MissingPreset', '%s has no original %s grid. Set config.tau or config.tau_by_method.%s explicitly.', config.dataset, method, key);
        end
        values = preset.(field);
    end
    validateattributes(values, {'numeric'}, {'row', 'nonempty', 'finite', '>=', 0, '<=', 1});
end
