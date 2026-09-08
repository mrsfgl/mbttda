function [images, labels, shape] = synthetic_dataset()
    % SYNTHETIC_DATASET Tiny labeled six-mode tensors; uses the caller's RNG stream.
    shape = 2 * ones(1, 6);
    classes = 3;
    samples = 12;
    templates = randn(prod(shape), classes);
    images = repelem(templates, 1, samples) + 0.15 * randn(prod(shape), classes * samples);
    labels = repelem(1:classes, samples);
end
