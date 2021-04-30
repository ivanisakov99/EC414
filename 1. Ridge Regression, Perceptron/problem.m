%% Problem 3.2: Ridge Regression
clear all
close all

load prostateStnd;

[n, d] = size(Xtrain);
W = zeros(d, 16);    % Store coefficient vectors
mseTrain = zeros(1, 16);    % Store mean squared errors
mseTest = zeros(1, 16);
lambda = 10^-5;

[Xtrain, Xtest] = normalize_data(Xtrain, Xtest);
[ytrain, ytest] = normalize_data(ytrain, ytest);
for i=1:16
    % Calculate coefficients and bias
    [w b] = train_rls(Xtrain, ytrain, lambda);
    W(:, i) = w;
    % Calculate MSE
    pred = Xtrain*w + b;
    mseTrain(i) = mean((pred - ytrain).^2);
    pred = Xtest*w + b;
    mseTest(i) = mean((pred - ytest).^2);
    lambda = lambda * 10;
end

% Show the first graph
figure(1)
for i=1:d
    plot([-5:10], W(i, :),'LineWidth',2.5);
    hold on
end
title("coefficient curves")
xlabel("lg(lambda)")
ylabel("w_i for")
legend({'lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45'})

% Save the plot
print -dpng Q3_1.png

% Show the second graph
figure(2)
plot([-5:10], mseTrain, 'LineWidth', 2.5);
hold on
plot([-5:10], mseTest, 'LineWidth', 2.5);
title("MSE curves")
xlabel("lg(lambda)")
ylabel("MSE on")
legend({'training set','testing set'})

% Save the plot
print -dpng Q3_2.png

%% Problem 3.4: Perceptron
clear all
close all

load adult_train_test

for k = 1:10
    % Shuffle training set
    idx = randperm(numel(ytrain));
    Xtrain = Xtrain(idx,:);
    ytrain = ytrain(idx);
    
    % Train percetron (1 pass over training set)
    [w,b,average_w,average_b] = train_perceptron(Xtrain,ytrain);
    
    % Test
    test_err_last_array(k) = numel(find(ytest ~= sign(Xtest * w + b))) / numel(ytest);
    test_err_average_array(k) = numel(find(ytest ~= sign(Xtest * average_w + average_b))) / numel(ytest);
end

for i = 1:10
    fprintf('test_err_last_array(%d): %d    ', i, test_err_last_array(i));
    fprintf('test_err_average_array(%d): %d\n', i, test_err_average_array(i));
end