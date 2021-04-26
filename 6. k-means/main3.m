%% Generate Gaussian Dataset
clear all;
close all;

mu = [[-2;-2], [-2;2], [2;-2], [2;2]];
sigma = [.5 .2 .3 .1];
X = generate_dataset(1000, mu, sigma);

%% Plot Dataset
figure;
plot(X(:,1), X(:,2), 'x', 'LineWidth', 2, 'Color', [0,0,1]);
xlabel('X_1', 'Fontsize', 20);
ylabel('X_2', 'Fontsize', 20);
title('Dataset', 'Fontsize', 20);
%% Run k-means
k = 4;
[y_hat, c] = kmeans(X_tr, k, 3); 

%Plot k-means
figure;
for i = 1:k
    index = find(y_hat == i);
    plot(X(index, 1), X(index, 2), 'x', 'Linewidth', 2);
    hold on;
end
plot(c(:, 1), c(:, 2), 'o', 'Linewidth', 4, 'Color', [0,1,0]);

%% Generate Transformed Dataset
dist = [1, 4, 10];
X = generate_dataset2(1000, dist);

%% Transform Data
X_tr = abs(X(:, 1)) + abs(X(:, 2));
plot(X_tr, zeros(size(X_tr)), 'x', 'Linewidth', 2);

%% Run k-means
k = 3;
[y_hat, c] = kmeans(X_tr, k, 3); 

%Plot k-means
figure;
for i = 1:k
    index = find(y_hat == i);
    plot(X(index, 1), X(index, 2), 'x', 'Linewidth', 2);
    hold on;
end
plot(c(:, 1), c(:, 2), 'o', 'Linewidth', 4, 'Color', [0,1,0]);


%% Functions
function X = generate_dataset(N, mu, sigma)
[d, k] = size(mu);

y = randi([1, k], [N, 1]);
X = [];

for i = 1:k
    n_k = sum(y == i);
    mu_k = mu(:, i);
    sigma_k = sigma(i) * eye(d);
    X = [X; mvnrnd(mu_k, sigma_k, n_k)];
end
end

function X = generate_dataset2(N, dist)
d = 2;
k = length(dist);

y = randi([1, k], [N, 1]);
X = [];

for i = 1:k
    n_k = sum(y == i);
    x_1 = 2 * dist(i) * rand(n_k, 1) - dist(i);
    x_2 = (randi([0, 1], [n_k, 1]) * 2 - 1) .* (dist(i) - abs(x_1));
    X = [X; [x_1, x_2] + mvnrnd(zeros(d, 1), .01 * eye(d), n_k)];
end
end
