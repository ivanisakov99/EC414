%% Example 1: Denoising with k-means
clear all;
close all;

X = imread('mandrill.jpg');
imshow(X);

X = im2double(X);
title('Original Image', 'Fontsize', 20);

[p1, p2, d] = size(X);
X = reshape(X, [p1 * p2, d]);

%% Change k to see the difference
k = 16; %Try different numbers like 2, 3, 8, 16, etc.
[y_hat, c] = kmeans(X, k, 1);

X_hat = zeros(size(X));

for i = 1:k
    index = find(y_hat == i);
    X_hat(index, :) = ones(length(index), 1) .* c(i, :);
end

X_hat = reshape(X_hat, [p1, p2, d]);
X_hat = uint8(256 * X_hat);
figure;
imshow(X_hat);
title(['k = ', int2str(k)], 'Fontsize', 20);

%% Denoising
figure;
X = imread('mandrill.jpg');
imshow(X);

figure;
X = reshape(X, [320 * 320, 3]);
X = im2double(X);
X = X + mvnrnd(zeros(3, 1), .100 * eye(3), 320 * 320);
X = reshape(X, 320, 320, 3);
X = uint8(256 * X);
imshow(X);
