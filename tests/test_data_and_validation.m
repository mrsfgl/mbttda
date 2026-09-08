function tests = test_data_and_validation
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.addTeardown(@path, path);
    testCase.addTeardown(@rng, rng);
    addpath(fullfile(fileparts(mfilename('fullpath')), 'helpers'));
    enter_test_environment();
end

function testSplitReproducibilityAndHoldout(testCase)
    [data, ~] = tiny_fixture();
    [pool, allocation] = reserve_validation(data, 2);
    rng(25);
    first = validation_fold(pool, 6);
    rng(25);
    second = validation_fold(pool, 6);
    testCase.verifyEqual(first.train_indices, second.train_indices);
    testCase.verifyEmpty(intersect(allocation.final_test, pool.train_indices));
    testCase.verifyEmpty(intersect(first.train_indices, first.test_indices));
    testCase.verifyEqual(sort([first.train_indices, first.test_indices]), sort(pool.train_indices));
    testCase.verifyEqual(numel(allocation.validation_reserved), 6);
    testCase.verifyError(@() reserve_validation(data, 6), 'mbttda:InsufficientValidation');
end

function testNonconsecutiveLabelsAndCompatibility(testCase)
    images = reshape(1:48, 2, 24);
    labels = repelem([10, 30, 90], 8);
    p = struct('classsize_list', 4, 'noise', 0);
    rng(17);
    data = get_data(images, labels, p, 1);
    rng(17);
    [train, trLbl, test, tsLbl] = get_data(images, labels, 3, 4, 0);
    testCase.verifyEqual(data.train, train);
    testCase.verifyEqual(data.trLbl, trLbl);
    testCase.verifyEqual(data.test, test);
    testCase.verifyEqual(data.tsLbl, tsLbl);
    checked = validate_dataset(data, 2);
    testCase.verifyEqual(unique(checked.trLbl), [10, 30, 90]);
    testCase.verifyError(@() validate_dataset(data, [2, 2]), 'mbttda:InvalidShape');
    data.trLbl(1) = 30;
    testCase.verifyError(@() validate_dataset(data, 2), 'mbttda:UnbalancedData');
end

function testMissingDataAndIncompletePresets(testCase)
    folder = tempname;
    testCase.verifyError(@() read_dataset_array('Cambridge', folder), 'mbttda:MissingData');
    p = dataset_preset('Weizmann');
    testCase.verifyEqual(p.tensor_shape, [8, 8, 44]);
    testCase.verifyEqual(p.nAbl, [0, 0, 0]);
    config = experiment_config('UCF101');
    testCase.verifyError(@() method_tau_grid('TTDA', config, dataset_preset('UCF101')), 'mbttda:MissingPreset');
    config.tau = 0.12;
    testCase.verifyEqual(method_tau_grid('TTDA', config, dataset_preset('UCF101')), 0.12);
end

function testTemporaryMatInput(testCase)
    folder = tempname;
    mkdir(folder);
    cleanup = onCleanup(@() rmdir(folder, 's')); %#ok<NASGU>
    Data = uint8(reshape(mod(1:64 * 44 * 4 * 2, 255), [64, 44, 4, 2])); %#ok<NASGU>
    save(fullfile(folder, 'WeizmannData.mat'), 'Data');
    [images, labels, preset] = load_data('Weizmann', folder);
    testCase.verifySize(images, [2816, 8]);
    testCase.verifyEqual(labels, [1, 1, 1, 1, 2, 2, 2, 2]);
    testCase.verifyEqual(preset.nSamp, 4);
end
