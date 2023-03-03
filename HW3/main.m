clear all;
close all;

if ~isempty(mfilename) && strcmp(mfilename, 'main')
    filename = fullfile(fileparts(mfilename('fullpath')), "flarsheim-1.jpg");

    image = Image(fullfile(filename), "", imread(filename));
    scales = [1.200 1.518 1.920 2.429 3.072 3.886 4.915];
    log_pyramid = get_LoG_pyramid(image, scales);

    figure;

    for i = 1:length(scales)
        subplot(2, 4, i);
        imshow(log_pyramid(:, :, i), []);
        title(sprintf("sigma = %f", scales(i)));
    end

end

function [LoG_pyramid] = get_LoG_pyramid(image, scales)
    LoG_pyramid = [];

    max_scale = max(scales);
    min_scale = min(scales);

    for i = 1:length(scales)
        sigma = scales(i);
        % LoG = fspecial('laplacian', (sigma - min_scale) / (max_scale - min_scale));
        LoG = fspecial('log', 11, sigma) % idk why hsize is 11, or what hsize is
        LoG_pyramid(:, :, i) = imfilter(image.grayscale_image, LoG, 'replicate');
    end

end

function [x0, y_max] = getScaleResponseExtrema(x, y)
    % Compute the LoG response using a polynomial fit
    p = polyfit(x, y, 3);
    f = polyval(p, x);

    % Compute the gradient of the LoG response
    df = gradient(f, x);

    % Find the extrema of the LoG response
    idx = find(df(1:end - 1) .* df(2:end) <= 0);
    x0 = x(idx);
    y_max = f(idx);
end
