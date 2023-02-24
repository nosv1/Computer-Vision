classdef Image

    properties
        path
        label
        rgb_image
        name
        hsv_image
        flattened_hsv_image
        grayscale_image
        sift_features
        hog_features
        dsift_features
        sift_fisher_vectors
        dsift_fisher_vectors
    end

    methods

        function obj = Image(path, label, rgb_image)
            obj.path = path;
            obj.label = label;
            obj.rgb_image = rgb_image;

            obj.name = strsplit(path, '/');
            obj.name = obj.name{end};
            obj.hsv_image = rgb2hsv(rgb_image);
            obj.flattened_hsv_image = reshape(obj.hsv_image, [], 3);
            obj.grayscale_image = rgb2gray(rgb_image);

            obj.sift_features = [];
            obj.hog_features = [];
            obj.dsift_features = [];

            obj.sift_fisher_vectors = {};
            obj.dsift_fisher_vectors = {};
        end

    end

end
