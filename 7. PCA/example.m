%% 3D example
clear all;
close all;

N = 1000;
X = mvnrnd([3, 2, 3], [3, 0.5, 2.5; 0.5, 3, 0.5; 2.5, 0.5, 3], N);
[T, V_tilde, mu, Lambda] = PCA(X, 2);
Lambda = 3 * Lambda/max(Lambda);

subplot(1, 2, 1)
plot3(X(:, 1), X(:, 2), X(:, 3), 'x', 'LineWidth', 2, 'Color', [0,0,1]);
%vectarrow(mu, Lambda(2) * V_tilde(:, 2) + mu');
%vectarrow(mu, Lambda(1) * V_tilde(:, 1) + mu');
xlabel('x_1', 'Fontsize', 20);
ylabel('x_2', 'Fontsize', 20);
zlabel('x_3', 'Fontsize', 20);
title('Dataset', 'Fontsize', 20);
hold on;



subplot(1, 2, 2)
plot(T(:, 1), T(:, 2), 'x', 'Color', [0,0,1]);
xlabel('v_1', 'Fontsize', 20);
ylabel('v_2', 'Fontsize', 20);
title('PCA with 2 Principal Components', 'FontSize', 20);

figure;
X_hat = T * V_tilde' + mu;
plot3(X_hat(:, 1), X_hat(:, 2), X_hat(:, 3), 'x', 'LineWidth', 2, 'Color', [0,0,1]);
xlabel('x_1', 'FontSize', 20);
ylabel('x_2', 'FontSize', 20);
zlabel('x_3', 'FontSize', 20);
title('Reconstructed Dataset', 'FontSize', 20);


%% Non-Linear Realtion between axis and PCA
X = 2 * rand(1000, 1);
X = [X, X.^2 + 0.5*randn(1000, 1)];

plot(X(:, 1), X(:, 2), 'x', 'LineWidth', 2);
xlabel('x_1', 'FontSize', 20);
ylabel('x_2', 'FontSize', 20);
title('Dataset', 'Fontsize', 20);

[T, V_tilde, mu, Lambda] = PCA(X, 2);
hold on;
%vecarrow(mu', 1*V_tilde(:, 2) + mu');
%vecarrow(mu', 0.5*V_tilde(:, 1) +mu');
xlim([0, 4]);
ylim([0, 4]);

%% MNIST Denoising 1
clear all

load('mnist.mat');

X = full(Xtr(1,:));
Xte = full(Xte);

mu = mean(Xte);
X_centralised = Xte - mu;
C = (X_centralised' * X_centralised);
[V, L] = eig(C);
[A,idx_sort]=sort(diag(L),'descend');
Lambda = diag(A);


plot(diag(Lambda), 'LineWidth', 2, 'Color', [0,0,1]);

V_tilde = V(:, idx_sort(1:784));
T = X_centralised * V_tilde;
X_hat = T * V_tilde' + mu;
X_hat = reshape(X_hat(1, :), 28, 28)';

subplot(1, 3, 1)
imagesc(X_hat);
title('784 PC', 'FontSize', 20);

V_tilde = V(:, idx_sort(1:392));
T = X_centralised * V_tilde;
X_hat = T * V_tilde' + mu;
X_hat = reshape(X_hat(1, :), 28, 28)';

subplot(1, 3, 2)
imagesc(X_hat);
title('392 PC', 'FontSize', 20);

V_tilde = V(:, idx_sort(1:56));
T = X_centralised * V_tilde;
X_hat = T * V_tilde' + mu;
X_hat = reshape(X_hat(1, :), 28, 28)';

subplot(1, 3, 3)
imagesc(X_hat);
title('56 PC', 'FontSize', 20);

%% MNIST Denoising 2
clear all

load('mnist.mat');
Xte = full(Xte);

X_noisy = Xte + 0.2 * randn(size(Xte));

mu = mean(X_noisy);
X_centralised = X_noisy - mu;
C = X_centralised' * X_centralised;
[V, L] = eig(C);
[A,idx_sort]=sort(diag(L),'descend');
Lambda = diag(A);

subplot(1, 2, 1);
imagesc(reshape(Xte(1, :), 28, 28)');
title('Clean Image', 'FontSize', 20);

subplot(1, 2, 2);
imagesc(reshape(X_noisy(1, :), 28, 28)');
title('Noisy Image', 'FontSize', 20);

%% MNIST Denoising 3

for i = 2:length(V)
    V_tilde = V(:, idx_sort(1:i));
    T = X_centralised * V_tilde;
    X_hat = T * V_tilde' + mu;
    MSE(i - 1) = mean(sum( (Xte - X_hat).^2, 2) );
end
[~, idx_min] = min(MSE);
figure;
plot(MSE, 'LineWidth', 2);
title('Denoising with PCA', 'FontSize', 20);
xlabel('Number of Principle Components', 'FontSize', 20);
ylabel('MSE', 'FontSize', 20);

figure;
subplot(1, 3, 1);
imagesc(reshape(Xte(1, :), 28, 28)');
title('Clean Image', 'FontSize', 20);

subplot(1, 3, 2);
imagesc(reshape(X_noisy(1, :), 28, 28)');
title('Noisy Image', 'FontSize', 20);

subplot(1, 3, 3);
imagesc(reshape(Xte(1, :), 28, 28)');
title([num2str(idx_min),' PC Image'], 'FontSize', 20);

%% Functions
function [T, V_tilde, mu, Lambda] = PCA(X, k)
mu = mean(X);
X_centralised = X - mu;
C = (X_centralised' * X_centralised);
[V, Lambda] = eig(C);
V_tilde = V(:, [end - k + 1:end]);
Lambda = diag(Lambda);
T = X_centralised * V_tilde;
end

