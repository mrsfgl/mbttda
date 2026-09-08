function [data, parameters] = tiny_fixture()
    % TINY_FIXTURE Nondegenerate, balanced classes and a small six-mode tensor.
    previous = rng;
    restore = onCleanup(@() rng(previous)); %#ok<NASGU>
    rng(41, 'twister');
    [images, labels, shape] = synthetic_dataset();
    preset = struct('classsize_list', 6, 'noise', 0, 'scale', 1);
    data = get_data(images, labels, preset, 1);
    parameters = algorithm_parameters(shape, struct('maxiter', 2, 'tau', 0.35));
end
