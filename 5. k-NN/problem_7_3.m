%% Problem 7.3: k-NN
clear all
close all

%% 7.3b
T = 21; % Must be odd

[X, y] = read_data; %Load Data

[m, ~] = size(X);
ntrain = round(m*0.60);
nval = round(m*0.20);
order = randperm(m);

Xtrain = X(order(1:ntrain),:);
ytrain = y(order(1:ntrain));

Xval = X(order(ntrain+1:ntrain+nval),:);
yval = y(order(ntrain+1:ntrain+nval));

Xtest = X(order(ntrain+nval+1:end),:);
ytest = y(order(ntrain+nval+1:end));

j = 1;
for i =1:2:T
y_hat = predict_knn(Xtrain, ytrain, Xval, i);
error(j) = sum(yval ~= y_hat)./length(yval);
j = j + 1;
end

figure;
plot(1:2:T, error, 'LineWidth', 2);
hold on
xlabel('k', 'FontSize', 20);
ylabel('Average 0/1 Validation Loss', 'FontSize', 20);
print -dpng Q7_3.png

%% 7.3c
[~, bestk] = min(error);
k = (bestk * 2) - 1;
fprintf('Best k: %d\n', k);

y_hat = predict_knn(Xtrain, ytrain, Xtest, k);
lowest_err = sum(ytest ~= y_hat)./length(ytest);
fprintf('Lowest error: %.2f\n', lowest_err);
