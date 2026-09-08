function result = main_CMDA(varargin)
    % MAIN_CMDA Compatibility entry point for either historical argument layout.
    setup_mbttda;
    [data, shape, ranks, neighbors] = unpack_mda_arguments(varargin{:});
    check_dependencies({'CMDA'});
    result = mda_classifier('CMDA', data, shape, ranks, neighbors);
    if nargin == 8
        result.Storage = result.metrics.factor_elements + result.metrics.feature_elements;
        result.metrics.legacy_storage_definition = 'Original eight-argument MDA interface: absolute factor plus feature element count';
    end
end
