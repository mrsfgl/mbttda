function info = setup_mbttda(externalPaths)
    % SETUP_MBTTDA Add project code and required bundled toolboxes to this session.
    % setup_mbttda({'C:/toolboxes/TT-Toolbox','C:/toolboxes/TTeMPS'}) also adds
    % optional dependencies. Does not call savepath or change the working directory.
    root = fileparts(mfilename('fullpath'));
    if nargin < 1
        externalPaths = {};
    end
    if ischar(externalPaths) || isstring(externalPaths)
        externalPaths = cellstr(externalPaths);
    end
    for i = 1:numel(externalPaths)
        if ~isfolder(externalPaths{i})
            error('mbttda:MissingDependency', 'Toolbox directory does not exist: %s', externalPaths{i});
        end
        addpath(genpath(externalPaths{i}), '-end');
    end
    bundled = {'FOptM', 'tensorlab', fullfile('tptool', 'array'), ...
               fullfile('tensor_classification-master', 'code', 'tensor_classification')};
    for i = 1:numel(bundled)
        addpath(fullfile(root, 'third-party', bundled{i}), '-end');
    end
    addpath(genpath(fullfile(root, 'src')), '-begin');
    addpath(root, '-begin');
    info.root = root;
    info.matlab = version;
    info.platform = computer;
    info.toolboxes = ver;
    info.original_revision = '189e39fa5e2d958616b389905f92b69f30f85ce4';
end
