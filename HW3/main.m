clear all;
close all;

if ~isempty(mfilename) && strcmp(mfilename, 'main')
    filename = fullfile(fileparts(mfilename('fullpath')), "flarsheim-1.jpg");

    image = Image(fullfile(filename), "", imread(filename));

    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % % Log Pyramid Stuff

    [height, width] = size(image.grayscale_image);
    scales = [1.200 1.518 1.920 2.429 3.072 3.886 4.915];
    n_scales = length(scales);

    a = 1.2;
    b = 2;
    log_max_scale = 11;

    log_pyramid = zeros(n_scales, height, width);

    for k = 1:n_scales
        % scales(k) = a * sqrt(b) ^ (k - 1);
        f_log{k} = fspecial('log', log_max_scale, scales(k));
        im_log{k} = imfilter(image.grayscale_image, f_log{k});
        log_pyramid(k, :, :) = im_log{k};
    end

    % % Figure for the log pyramid
    % figure;

    % for k = 1:n_scales
    %     subplot(2, 4, k);
    %     imshow(im_log{k}, []);
    %     title(sprintf('LoG, scale = %f', scales(k)));
    % end

    % end figure

    % % End Log Pyramid Stuff
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %

    log_peak_threshold = 10;
    peak = 16;
    log_gradient_threshold = 0.2;

    log_max = reshape(max(reshape(log_pyramid(:, :, :), [n_scales, height * width])), [height, width]);
    alp_offs = find(log_max >= log_peak_threshold);
    n_peak_log = length(alp_offs);

    log_sift_pos = zeros(size(log_max));
    n_log_sift = 0;

    for k = 1:length(alp_offs)
        [alp_i, alp_j] = ind2sub(size(log_max), alp_offs(k));
        [x0_log(k), peak_response_log(k), y0(k)] = getScaleResponseExtrema(scales, log_pyramid(:, alp_i, alp_j)');

        if abs(y0(k)) >= log_gradient_threshold
            continue;
        end

        log_sift_pos(alp_i, alp_j) = 1;
        n_log_sift = n_log_sift + 1;
        k_pos(n_log_sift, 1) = alp_j;
        k_pos(n_log_sift, 2) = alp_i;
        k_scale(n_log_sift) = x0_log(k);
        k_grad(n_log_sift) = y0(k);
    end

    % % Figure for the log sift

    % figure(1); imshow(image.grayscale_image, []); hold on;
    % plot(k_pos(:, 1), k_pos(:, 2), '*r');
    % title(sprintf('LoG SIFT keypoints (threshold = %d)', log_peak_threshold));

    % % end figure

end

function [x0, peak_response, y0] = getScaleResponseExtrema(x, y)
    [p, s] = polyfit(x, y, 3); % a*x^3 + b*x^2 + c*x + d
    delta = 0.2;
    x_ = [x(1):delta:x(end)];
    fx = polyval(p, x_);
    [peak_response, peak_index] = max(fx);
    x0 = x_(peak_index); % x0 is the scale at which the response is maximum
    y0 = 3 * p(1) * x0 ^ 2 + 2 * p(2) * x0 + p(3); % y0 is the maximum response
    y0_ = 6 * p(1) * x0 + 2 * p(2); % y0_ is the first derivative of the response at x0

    x1 = (-2 * p(2) - sqrt(4 * p(2) * p(2) - 12 * p(1) * p(3))) / 6 * p(1);
    x2 = (-2 * p(2) + sqrt(4 * p(2) * p(2) - 12 * p(1) * p(3))) / 6 * p(1);

    % figure(41); subplot(1, 2, 1); hold on; grid on; stem(x, y, '+b');
    % plot(x_, fx, ':b'); stem(x0, peak_response, '*r');
    % title('LoG response'); xlabel('\sigma'); ylabel('L(x,y)');
    % subplot(1, 2, 2); hold on; grid on;
    % plot(x_, 3 * p(1) * x_ .^ 2 + 2 * p(2) * x_ + p(3), ':b'); plot(x0, y0, '*r');
    % title('LoG response diff'); xlabel('\sigma');

end
