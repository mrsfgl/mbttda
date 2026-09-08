function result = main_DGTDA(varargin)
    % MAIN_DGTDA Compatibility entry point for either historical argument layout.
    setup_mbttda;
    [data, shape, ranks, neighbors] = unpack_mda_arguments(varargin{:});
    check_dependencies({'DGTDA'});
    result = mda_classifier('DGTDA', data, shape, ranks, neighbors);
    if nargin == 8
        result.Storage = result.metrics.factor_elements + result.metrics.feature_elements;
        result.metrics.legacy_storage_definition = 'Original eight-argument MDA interface: absolute factor plus feature element count';
    end
end
