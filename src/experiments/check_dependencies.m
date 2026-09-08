function missing = check_dependencies(methods, failIfMissing)
    % CHECK_DEPENDENCIES Report prerequisites for only the selected methods.
    if nargin < 2
        failIfMissing = true;
    end
    methods = cellstr(upper(string(methods)));
    required = {'fitcknn'};
    for i = 1:numel(methods)
        switch methods{i}
            case {'TTDA', 'TWTTDA', 'THW', '2WTTDA', '3WTTDA'}
                required = [required, {'OptStiefelGBB', 'tmprod', 'tens2mat'}]; %#ok<AGROW>
            case 'MPS'
                required = [required, {'tmprod', 'tens2mat'}]; %#ok<AGROW>
            case {'BTT', 'TWBTT', 'THWBTT'}
                required = [required, {'tt_matrix', 'TTeMPS_op', 'amen_eigenvalue', 'lobpcg', 'tmprod'}]; %#ok<AGROW>
            case 'TTNPE'
                required = [required, {'OptStiefelGBB', 'knnsearch', 'tens2mat', ...
                                       'TenConPro', 'L', 'R', 'L_inv', 'LSQ_Unitary_L', 'LSQ_Unitary_R', 'AXB', 'XAB'}]; %#ok<AGROW>
            case {'CMDA', 'DGTDA'}
                required = [required, {methods{i}, 'classbased_differences', 'tmprod'}]; %#ok<AGROW>
            case 'LDA'
                % Dense LDA only needs the classifier toolbox.
            otherwise
                error('mbttda:UnknownMethod', 'Unknown method: %s', methods{i});
        end
    end
    required = unique(required, 'stable');
    missing = required(cellfun(@(name) isempty(which(name)), required));
    if failIfMissing && ~isempty(missing)
        error('mbttda:MissingDependency', ...
              'Missing: %s. See docs/dependencies.md and setup_mbttda(externalPaths).', strjoin(missing, ', '));
    end
end
