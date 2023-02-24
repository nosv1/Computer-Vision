%Yogurt
close all;
clear all;

% sift = 'SIFT-on-MATLAB';
% addpath(genpath([pwd, filesep, sift]));

LABELS = {'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach', 'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud', 'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway', 'golf_course', 'ground_track_field', 'harbor', 'industrial_area', 'intersection', 'island', 'lake', 'meadow', 'medium_residential', 'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot', 'railway', 'railway_station', 'rectangular_farmland', 'river', 'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg', 'sparse_residential', 'stadium', 'storage_tank', 'tennis_court', 'terrace', 'thermal_power_station', 'wetland'};

LABEL_SETS = {[19, 18, 21, 42, 11, 37, 25, 16, 10, 36], [39, 33, 43, 29, 16, 8, 5, 32, 44, 42], [31, 3, 6, 41, 22, 45, 17, 38, 25, 39], [7, 14, 8, 35, 5, 45, 9, 4, 18, 15], [16, 36, 22, 29, 4, 43, 11, 23, 13, 8], [40, 7, 19, 4, 34, 23, 16, 24, 13, 44], [45, 10, 29, 31, 27, 4, 7, 44, 37, 23], [14, 40, 32, 38, 21, 11, 22, 18, 2, 3], [7, 26, 19, 37, 39, 1, 9, 25, 43, 5], [21, 1, 20, 37, 4, 30, 17, 19, 42, 18]};

if ~isempty(mfilename) && strcmp(mfilename, 'main')
    nwpu_path = fullfile(fileparts(mfilename('fullpath')), '../NWPU-RESISC45/');

    USER_ID = '16340679';

    labels = LABELS(LABEL_SETS{str2double(USER_ID(end))});

    TRAIN_GMMS = true;
    LOAD_IMAGES_FROM_FILE = false;
    COMPUTE_FEATURES = true;
    PLOT_A_COLUMNS = false;
    COMPUTE_FISHER_VECTORS = true;
    KNN_BENCHMARKS = true;
    NUM_IMAGES_PER_LABEL = 20;

    images = load_images(nwpu_path, labels, NUM_IMAGES_PER_LABEL, LOAD_IMAGES_FROM_FILE);

    f_sift = {};
    f_dsift = {};

    if COMPUTE_FEATURES

        for i = 1:numel(labels)
            label = labels{i};
            label_images = images(label);
            fprintf('Processing %s images...\n', label);

            for j = 1:numel(images(label))
                image = label_images(j);

                [x, sift_features] = get_image_features(single(image.grayscale_image), ImageFeatureOperations.SIFT);
                f_sift = [f_sift, sift_features];
                image.sift_features = sift_features;

                [x, dsift_features] = get_image_features(single(image.grayscale_image), ImageFeatureOperations.DSIFT);
                f_dsift = [f_dsift, dsift_features];
                image.dsift_features = dsift_features;

                label_images(j) = image;
            end

            images(label) = label_images;

        end

        fprintf('Saving features...\n')
        save('images.mat', 'images', '-v7.3')

    else
        fprintf('Loading features...\n')

        for i = 1:numel(labels)
            label = labels{i};

            for j = 1:numel(images(label))
                label_images = images(label);
                image = label_images(j);
                f_sift = [f_sift, image.sift_features];
                f_dsift = [f_dsift, image.dsift_features];
            end

        end

    end

    f_sift_flattened = cell2mat(datasample(f_sift, 20))';
    f_dsift_flattened = cell2mat(datasample(f_dsift, 20))';

    kds = [16, 32];
    ncs = [32, 64, 128];

    if TRAIN_GMMS
        [sift_A, sift_gmm] = train_sift_gmm(f_sift_flattened, kds, ncs, ImageFeatureOperations.SIFT);
        [dsift_A, dsift_gmm] = train_sift_gmm(f_dsift_flattened, kds, ncs, ImageFeatureOperations.DSIFT);
    else
        fprintf('Loading SIFT GMM...\n');
        load('sift_gmm.mat', 'sift_A', 'sift_gmm');
        fprintf('Loading dense SIFT GMM...\n');
        load('dsift_gmm.mat', 'dsift_A', 'dsift_gmm');
    end

    if PLOT_A_COLUMNS
        figure;
        sgtitle('SIFT A First 6 Columns');

        for i = 1:6
            subplot(2, 3, i);
            plot(sift_A(:, i));
            title(sprintf('Column %d', i));
        end

        figure;
        sgtitle('Dense SIFT A First 6 Columns');

        for i = 1:6
            subplot(2, 3, i);
            plot(dsift_A(:, i));
            title(sprintf('Column %d', i));
        end

    end

    if COMPUTE_FISHER_VECTORS

        for i = 1:numel(labels)
            label = labels{i};
            label_images = images(label);
            fprintf('Getting fisher vectors %s images...\n', label);

            for j = 1:numel(images(label))
                image = label_images(j);

                image.sift_fisher_vectors = {};
                image.dsift_fisher_vectors = {};

                for k = 1:size(sift_gmm, 1)

                    for m = 1:size(sift_gmm, 2)
                        sift_fisher_vector = get_fisher_vector(image.sift_features', sift_A, sift_gmm(k, m), kds(k), ncs(m));
                        dsift_fisher_vector = get_fisher_vector(image.dsift_features', dsift_A, dsift_gmm(k, m), kds(k), ncs(m));
                        image.sift_fisher_vectors = [image.sift_fisher_vectors, sift_fisher_vector];
                        image.dsift_fisher_vectors = [image.dsift_fisher_vectors, dsift_fisher_vector];
                    end

                end

                label_images(j) = image;

            end

            images(label) = label_images;

        end

        fprintf('Saving fisher vectors...\n')
        save('images.mat', 'images', '-v7.3')
    end

    sift_benchmarks = [];
    dsift_benchmarks = [];

    if KNN_BENCHMARKS

        for i = 1:size(sift_gmm, 1)

            for j = 1:size(sift_gmm, 2)
                fprintf('Computing benchmarks for k = %d, nc = %d...\n', kds(i), ncs(j));

                offset = (i - 1) * size(sift_gmm, 1) + j;

                relevant_sift_fisher_vectors = {};
                relevant_dsift_fisher_vectors = {};
                relevant_labels = {};

                for k = 1:numel(labels)
                    label = labels{k};
                    label_images = images(label);

                    for l = 1:numel(label_images)
                        image = label_images(l);
                        relevant_sift_fisher_vectors = [relevant_sift_fisher_vectors, image.sift_fisher_vectors(offset)];
                        relevant_dsift_fisher_vectors = [relevant_dsift_fisher_vectors, image.dsift_fisher_vectors(offset)];
                        relevant_labels = [relevant_labels, image.label];
                    end

                end

                sift_benchmark = knn_benchmark(cell2mat(relevant_sift_fisher_vectors)', relevant_labels', kds(i), ncs(j));
                dsift_benchmark = knn_benchmark(cell2mat(relevant_dsift_fisher_vectors)', relevant_labels', kds(i), ncs(j));
                sift_benchmarks = [sift_benchmarks, sift_benchmark];
                dsift_benchmarks = [dsift_benchmarks, dsift_benchmark];

            end

        end

        fprintf('Saving benchmarks...\n')
        save('sift_benchmarks.mat', 'sift_benchmarks', '-v7.3');
        save('dsift_benchmarks.mat', 'dsift_benchmarks', '-v7.3');

    else
        fprintf('Loading SIFT benchmarks...\n');
        load('sift_benchmarks.mat', 'sift_benchmarks');
        fprintf('Loading dense SIFT benchmarks...\n');
        load('dsift_benchmarks.mat', 'dsift_benchmarks');
    end

end

function images = load_images(root, labels, num_images_per_label, from_file)

    if from_file
        fprintf('Loading images from file...\n');

        load('images.mat', 'images');
        return;
    end

    fprintf('Loading images from %s...\n', root);
    folders = dir(root);
    folders = {folders([folders.isdir]).name};
    folders = setdiff(folders, {'.', '..'});
    images = containers.Map();

    for i = 1:numel(labels)
        label = labels{i};

        if ~any(strcmp(folders, label))
            continue;
        end

        fprintf('Loading %s images...\n', label);
        images(label) = [];

        files = dir(fullfile(root, label, '*.jpg'));
        files = {files.name};

        for j = 1:num_images_per_label
            image_idx = randi(numel(files));
            % yes, it's possible this will pick the same image twice, cba to fix rn
            image = files{image_idx};
            img = imread(fullfile(root, label, image));
            images(label) = [images(label), Image(fullfile(root, label, image), label, img)];
        end

    end

    fprintf('Saving images...\n');
    save('images.mat', 'images', '-v7.3');
end

function [keypoints, descriptors] = get_image_features(image, operation)

    if strcmp(operation, ImageFeatureOperations.SIFT)
        [keypoints, descriptors] = vl_sift(image);
    elseif strcmp(operation, ImageFeatureOperations.DSIFT)
        [keypoints, descriptors] = vl_dsift(image);
    else
        error('Invalid operation');
    end

end

function [A, sift_gmm] = train_sift_gmm(sift_train_gmm, kds, ncs, operation)
    dbg = 0;

    if dbg
        data = 'data';
        addpath(genpath([pwd, filesep, data]));
        load([data, filesep, 'sift_train_gmm.mat']);
        kds = [8 12 16 24 32];
        ncs = [16, 24, 32, 64, 128];
    end

    tic;
    [A, s, lat] = pca(double(sift_train_gmm));
    t1 = toc;

    for j = 1:length(kds)
        x = double(sift_train_gmm) * A(:, 1:kds(j)); % data projected

        for k = 1:length(ncs)
            fprintf('\n t=%1.2f training gmm(%d %d)', toc, kds(j), ncs(k));
            [sift_gmm(j, k).mean, sift_gmm(j, k).cov, sift_gmm(j, k).prior, sift_gmm(j, k).ll] = vl_gmm(x', ncs(k), 'MaxNumIterations', 200);
            fprintf(' ll = %1.2f ', sift_gmm(j, k).ll);
        end % for k

    end % for j

    if strcmp(operation, ImageFeatureOperations.SIFT)
        fprintf('Saving SIFT GMM...\n')
        sift_A = A;
        save('sift_gmm.mat', 'sift_A', 'sift_gmm', '-v7.3');
    elseif strcmp(operation, ImageFeatureOperations.DSIFT)
        fprintf('Saving dense SIFT GMM...\n')
        dsift_gmm = sift_gmm;
        dsift_A = A;
        save('dsift_gmm.mat', 'dsift_A', 'dsift_gmm', '-v7.3');
    else
        error('Invalid operation');
    end

    % if dbg
    %     figure(41); title('SIFT GMM (8, 16)'); grid on; hold on;
    %     plot3d(sift_gmm(1, 1).mean', '*r');
    % end

    return;
end

function [fisher_vector] = get_fisher_vector(features, A, gmm, kd, nc)
    f0 = double(features) * A(:, 1:kd);
    fisher_vector = vl_fisher(f0', gmm.mean, gmm.cov, gmm.prior);
end

function [accuracy, cfmatrix] = knn_benchmark(features, labels, kd, nc)

    random_indicies = randperm(size(features, 1));
    training_features = features(random_indicies(1:floor(0.8 * size(features, 1))), :);
    training_labels = labels(random_indicies(1:floor(0.8 * size(features, 1))));
    test_features = features(random_indicies(floor(0.8 * size(features, 1)):end), :);
    test_labels = labels(random_indicies(floor(0.8 * size(features, 1)):end));

    % Train KNN model
    knn_model = fitcknn(training_features, training_labels);
    knn_predictions = predict(knn_model, test_features);
    accuracy = sum(strcmp(test_labels, knn_predictions)) / numel(test_labels);

    % Create confusion matrix
    figure;
    cfmatrix = confusionchart(test_labels, knn_predictions);
    cfmatrix.Title = sprintf('Confusion Matrix (KNN, kd=%d, nc=%d, accuracy=%f)', kd, nc, accuracy);
end
