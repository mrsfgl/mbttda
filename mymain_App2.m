function result = mymain_App2(varargin)
    % MYMAIN_APP2 Historical two-branch entry point.
    setup_mbttda;
    result = legacy_tt_call('twttda', varargin{:});
end
