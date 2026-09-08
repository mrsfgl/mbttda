function tests = test_tensor_math
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.addTeardown(@path, path);
    testCase.addTeardown(@rng, rng);
    addpath(fullfile(fileparts(mfilename('fullpath')), 'helpers'));
    enter_test_environment();
end

function testContractionAndReconstruction(testCase)
    cores = {reshape(1:4, [1, 2, 2]), reshape(1:6, [2, 3, 1])};
    expected = reshape(cores{1}, 2, 2) * reshape(cores{2}, 2, 3);
    testCase.verifyEqual(merge_tensor(cores), expected);
    tensor = reshape(sin(1:60), [2, 3, 2, 5]);
    decomposition = U2Ui_tau(tensor, 0);
    testCase.verifyEqual(reshape(merge_tensor(decomposition, false), size(tensor)), tensor, 'AbsTol', 1e-12);
    for mode = 1:numel(decomposition) - 1
        core = decomposition{mode};
        basis = reshape(core, [], size(core, 3));
        testCase.verifyEqual(basis' * basis, eye(size(core, 3)), 'AbsTol', 1e-12);
    end
end

function testThresholdBoundaryAndTwoDimensionalInput(testCase)
    [basis, values, nextValue] = svdtrunc2(diag([4, 2, 1]), 0.5);
    testCase.verifySize(basis, [3, 2]);
    testCase.verifyEqual(values, [4; 2]);
    testCase.verifyEqual(nextValue, 1);
    matrix = reshape(1:12, 3, 4);
    cores = U2Ui_tau(matrix, 0);
    testCase.verifyEqual(reshape(merge_tensor(cores, false), 3, 4), matrix, 'AbsTol', 1e-12);
    testCase.verifyEqual(wwq_squeeze(reshape(1:3, [1, 3, 1])), (1:3)');
end

function testBalancedPartitions(testCase)
    testCase.verifyEqual(computeK(2 * ones(1, 6), 2), 3);
    testCase.verifyEqual(computeK(2 * ones(1, 6), 3), [2, 4]);
    testCase.verifyEqual(computeK([3, 4, 5], 3), [1, 2]);
    testCase.verifyEmpty(computeK([3, 4], 1));
    testCase.verifyError(@() computeK([3, 4], 3), 'mbttda:InvalidBranches');
end

function testCoreQuadraticAgainstDenseMap(testCase)
    rng(12, 'twister');
    cases = {{randn(1, 2, 2), randn(2, 3, 2), randn(2, 2, 2)}, ...
             {randn(1, 2, 1), randn(1, 2, 1), randn(1, 2, 1)}};
    for fixture = cases
        cores = fixture{1};
        projection = dense_tt_reference(cores);
        scatter = randn(size(projection, 1));
        scatter = scatter + scatter';
        testCase.verifyEqual(ObjVal(scatter, cores), trace(projection' * scatter * projection), 'AbsTol', 1e-10);
        for mode = 1:numel(cores)
            map = zeros(numel(projection), numel(cores{mode}));
            for entry = 1:numel(cores{mode})
                unit = zeros(size(cores{mode}));
                unit(entry) = 1;
                perturbed = cores;
                perturbed{mode} = unit;
                dense = dense_tt_reference(perturbed);
                map(:, entry) = dense(:);
            end
            expected = map' * kron(eye(size(projection, 2)), scatter) * map;
            actual = core_quadratic(scatter, cores, mode);
            if mode == numel(cores)
                actual = kron(eye(size(cores{end}, 3)), actual);
            end
            testCase.verifyEqual(actual, expected, 'AbsTol', 1e-9);
        end
    end
end

function testBranchProjectionOrder(testCase)
    rng(9, 'twister');
    left = randn(2, 2);
    middle = randn(3, 2);
    right = randn(2, 1);
    samples = randn(2, 5, 3, 2);
    actual = project_branch_features(samples, {left, middle, right});
    expected = zeros(4, 5);
    for observation = 1:5
        features = zeros(2, 2, 1);
        for l = 1:2
            for m = 1:2
                value = 0;
                for i = 1:2
                    for j = 1:3
                        for k = 1:2
                            value = value + left(i, l) * middle(j, m) * right(k) * samples(i, observation, j, k);
                        end
                    end
                end
                features(l, m, 1) = value;
            end
        end
        ordered = permute(features, [2, 3, 1]);
        expected(:, observation) = ordered(:);
    end
    testCase.verifyEqual(actual, expected, 'AbsTol', 1e-12);
end
