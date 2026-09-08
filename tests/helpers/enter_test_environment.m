function enter_test_environment()
    % ENTER_TEST_ENVIRONMENT Install the checkout temporarily for an isolated suite.
    root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    addpath(root);
    setup_mbttda;
end
