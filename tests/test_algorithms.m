classdef test_algorithms < matlab.unittest.TestCase
    properties (TestParameter)
        method = {'TTDA', 'TWTTDA', 'THW', 'LDA', 'MPS', 'CMDA', 'DGTDA', 'TTNPE', 'BTT', 'TWBTT', 'THWBTT'}
    end
    methods (TestClassSetup)

        function setupPaths(testCase)
            testCase.addTeardown(@path, path);
            testCase.addTeardown(@rng, rng);
            addpath(fullfile(fileparts(mfilename('fullpath')), 'helpers'));
            enter_test_environment();
        end

    end
    methods (Test)

        function smoke(testCase, method)
            missing = check_dependencies({method}, false);
            if ismember(method, {'TTNPE', 'BTT', 'TWBTT', 'THWBTT'})
                testCase.assumeEmpty(missing, ['Optional dependency unavailable: ', strjoin(missing, ', ')]);
            else
                testCase.assertEmpty(missing, ['Required dependency unavailable: ', strjoin(missing, ', ')]);
            end
            [data, p] = tiny_fixture();
            result = run_method(method, data, p, ones(size(p.I)));
            predictions = result.PreLabel;
            if iscell(predictions)
                predictions = predictions{end};
            end
            testCase.verifySize(predictions, [size(data.test, 2), 1]);
            testCase.verifyGreaterThanOrEqual(result.PreErr, zeros(size(result.PreErr)));
            testCase.verifyLessThanOrEqual(result.PreErr, ones(size(result.PreErr)));
            testCase.verifyTrue(isfinite(result.metrics.storage_ratio));
            testCase.verifyGreaterThan(result.metrics.storage_ratio, 0);
            testCase.verifyGreaterThanOrEqual(sum(result.time_subspace), 0);
            timings = [result.time_subspace, result.time_embedding, result.time_classify];
            testCase.verifyTrue(all(isfinite(timings)));
            testCase.verifyTrue(all(ismember(predictions, data.trLbl)));
        end

    end
end
