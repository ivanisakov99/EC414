%% Example
clear all;
close all;

[X, y] = generate_dataset();
plot_dataset(X, y);

y_hat = kNN(X([100:200], :), X, y, 3, 'LOOCV');
% y_hat = kNN(X([100:200], :), X, y, 3);
acc = sum(y == y_hat)./length(y);

figure;
plot([1:10:length(y_hat(1,:))*10], acc, 'LineWidth', 2);
xlabel('k', 'FontSize', 20);
ylabel('Accuracy', 'FontSize', 20);

%% 
function [X, y] = generate_dataset()
N = 100;

mu_1 = [-3; -2];
mu_2 = [0; 0];
mu_3 = [-1; 2];

sigma_1 = [1.5, -0.2; -0.2, 1.5];
sigma_2 = [1, 0.5; 0.5, 1];
sigma_3 = [0.5, -0.1; -0.1, 0.5];

X_1 = mvnrnd(mu_1, sigma_1, N);
X_2 = mvnrnd(mu_2, sigma_2, N);
X_3 = mvnrnd(mu_3, sigma_3, N);

y_1 = 1 * ones(N, 1);
y_2 = 2 * ones(N, 1);
y_3 = 3 * ones(N, 1);

X = [X_1; X_2; X_3];
y = [y_1; y_2; y_3];

index = randperm(length(y));
X = X(index, :);
y = y(index, :);

end

function plot_dataset(X, y)
categories = unique(y);
plots = [];
legends = {};

for i = 1:length(categories)
    index = find(y == categories(i));
    p1 = plot(X(index, 1), X(index, 2), 'x', 'LineWidth', 2);
    hold on;
    plots = [plots, p1];
    legends{i} = num2str(categories(i));
end

title('Dataset', 'FontSize', 20);
xlabel('x_1', 'FontSize', 20);
ylabel('x_2', 'FontSize', 20);
legend(plots, legends, 'FontSize', 20);
end

function y_hat = kNN(X_test, X, y, k, str)
    [m1, d1] = size(X);
    [m2, d2] = size(X_test);

    if nargin < 5
        X = reshape(X, 1, m1, d1);
        X_test = reshape(X_test, m2, 1, d2);
        distance = sum((X - X_test).^2, 3);
        [~, nn_idx] = mink(distance, k, 2);
        nn_y = y(nn_idx);
        y_hat = mode(nn_y, 2);
        return
    end

    if(str == 'LOOCV')
        X = reshape(X, 1, m1, d1);
        X_test = reshape(X, m1, 1, d1);
        distance = sum((X - X_test).^2, 3);
        [~, nn_idx] = sort(distance, 2, 'ascend');
        nn_idx = nn_idx(:, 2:end);
        nn_y = y(nn_idx);
        for i = 1:length(nn_idx(1,:))/10
            y_hat(:, i) = mode(nn_y(:, [1:i*10]), 2);
        end
    end
end

