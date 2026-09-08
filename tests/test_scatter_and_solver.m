function tests = test_scatter_and_solver
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.addTeardown(@path, path);
    testCase.addTeardown(@rng, rng);
    addpath(fullfile(fileparts(mfilename('fullpath')), 'helpers'));
    enter_test_environment();
end

function testHandCalculatedScatter(testCase)
    data = zeros(2, 2, 2);
    data(:, :, 1) = [0, 2; 0, 0];
    data(:, :, 2) = [4, 6; 2, 2];
    [within, between] = const_scat(data);
    testCase.verifyEqual(within, [4, 0; 0, 0], 'AbsTol', 1e-14);
    testCase.verifyEqual(between, [8, 4; 4, 2], 'AbsTol', 1e-14);
    [legacyWithin, legacyBetween] = const_scat(data, 2, 2, 2);
    testCase.verifyEqual(legacyWithin, within);
    testCase.verifyEqual(legacyBetween, between);
end

function testProjectedSingletonAndStorageAccounting(testCase)
    data = reshape(1:12, [1, 2, 3, 2]);
    [within, between] = const_scat(data);
    testCase.verifySize(within, [1, 1]);
    testCase.verifySize(between, [1, 1]);
    metrics = result_metrics({zeros(1, 2, 2), zeros(2, 3, 1)}, zeros(2, 5), 100);
    testCase.verifyEqual(metrics.factor_elements, 10);
    testCase.verifyEqual(metrics.feature_elements, 10);
    testCase.verifyEqual(metrics.storage_ratio, 0.2);
end

function testSolverOutputOrderAndOrthogonality(testCase)
    [data, p] = tiny_fixture();
    cores = U2Ui_tau(reshape(data.train, [p.I, size(data.train, 2)]), p.tau);
    cores = cores(1:end - 1);
    folded = reshape(data.train, prod(p.I), 6, 3);
    [learned, objective, scatterSeconds, searchSeconds] = lrnU(folded, cores, p);
    testCase.verifyGreaterThanOrEqual(numel(objective), 2);
    testCase.verifyTrue(all(isfinite(objective)));
    testCase.verifySize(scatterSeconds, [1, 1]);
    testCase.verifySize(searchSeconds, [1, 1]);
    testCase.verifyGreaterThanOrEqual([scatterSeconds, searchSeconds], [0, 0]);
    [within, between] = const_scat(folded);
    testCase.verifyEqual(objective(end), ObjVal(within - p.lambda * between, learned), 'AbsTol', 1e-8);
    for core = learned
        matrix = reshape(core{1}, [], size(core{1}, 3));
        testCase.verifyEqual(matrix' * matrix, eye(size(matrix, 2)), 'AbsTol', 1e-5);
    end
end

function testSingleModeBttEigenvectors(testCase)
    data = reshape([0, 1, 4, 5, 1, 0, 5, 4], [2, 2, 2]);
    p = algorithm_parameters(2);
    [cores, seconds] = lrnU_btt(data, {reshape([1; 0], 1, 2, 1)}, p);
    [within, between] = const_scat(data);
    eigenvalues = eig(within - between);
    testCase.verifyEqual(ObjVal(within - between, cores), min(eigenvalues), 'AbsTol', 1e-12);
    testCase.verifyGreaterThanOrEqual(seconds, 0);
end

function testSmallestEigenvectorsIncludingFullRank(testCase)
    matrix = diag([4, -2, 1]);
    vectors = smallest_eigenvectors(matrix, 2);
    testCase.verifyEqual(diag(vectors' * matrix * vectors), [-2; 1], 'AbsTol', 1e-12);
    fullBasis = smallest_eigenvectors(matrix, 3);
    testCase.verifyEqual(diag(fullBasis' * matrix * fullBasis), [-2; 1; 4], 'AbsTol', 1e-12);
end

function testSingleModeTtnpeApproximation(testCase)
    scatter = diag([4, -2, 1]);
    p = algorithm_parameters(3);
    [cores, target, objective, approximationSeconds, eigenSeconds] = ...
        Apro_Solver({reshape([1; 0; 0], 1, 3, 1)}, scatter, p, optimizer_options());
    testCase.verifyEqual(reshape(cores{1}, 3, 1), target);
    testCase.verifyEqual(objective(end), -2, 'AbsTol', 1e-12);
    testCase.verifyGreaterThanOrEqual([approximationSeconds, eigenSeconds], [0, 0]);
end
