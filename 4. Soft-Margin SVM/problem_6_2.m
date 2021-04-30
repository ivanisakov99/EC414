%% Problem 6.2: Soft-Margin Binary Kernel SVM
close all
clear

load('kernel-svm-2rings.mat');
[m, n] = size(X);
T = 10000;
c = 100;
gamma = 2;

%% 6.2a
[alpha, b, obj, zeroOneAverageLoss] = train_ksvm_sd(X, y, T, c, gamma);

%% 6.2b
figure1 = figure;
plot(obj,'LineWidth',2,'Color', [0,0,1]);
title('RBF Learning Curve', 'FontSize', 20);
xlabel('Iteration', 'FontSize', 20);
ylabel('Objective Function', 'FontSize', 20);
print -dpng Q6_2b.png

%% 6.2c
figure2 = figure;
plot(zeroOneAverageLoss,'LineWidth',2,'Color', [0,0,1]);
title('0/1 Loss', 'FontSize', 20);
xlabel('Iteration', 'FontSize', 20);
ylabel('Average 0/1 Loss', 'FontSize', 20);
print -dpng Q6_2c.png

%% 6.2d
q = zeroOneAverageLoss(10000, 1);
display(q)

%% 6.2e
x_1 = [-4:.1:4]'.*ones(1,201);
x_2 = [-4:.1:4].*ones(201,1);
x_1 = x_1(:);
x_2 = x_2(:);
Xtest = [x_1, x_2];

K_test_tilde = [ones(1,length(Xtest(:,1))); Kernel(X, Xtest, gamma)];
alpha_tilde = [b; alpha];
val = sign(alpha_tilde'*K_test_tilde);
index1 = find(val==1);
index2 = find(val==-1);

figure3 = figure;
p1 = plot(x_1(index1), x_2(index1), '*', 'Color', [0,1,0]);
hold on;
p2 = plot(x_1(index2), x_2(index2), '*', 'Color', [1,0,0]);

index1 = find(y==1);
index2 = find(y==-1);
p3 = plot(X(index1,1), X(index1,2), 'x', 'Color', [0,0,1], 'LineWidth', 2);
p4 = plot(X(index2,1), X(index2,2), 'o', 'Color', [0,0,0], 'LineWidth', 2);
legend([p1, p2, p3, p4], {'$\hat{y} = 1$', '$\hat{y} = 1$', '$\hat{y} = 1$', '$\hat{y} = 1$'}, 'Interpreter', 'Latex', 'FontSize', 20);
print -dpng Q6_2e.png
