%% Problem 4.3: Gradient Descent for Logistic Regression

%% 4.3b

% Close all plots
close all
% Clear the workspace
clear   

% Load training and test data
load adult_train_test

% Get number of training samples and their dimension
[m,d] = size(Xtrain);

eta = 1/5000;
T = 1000;

for i = 1:10
    % Generate a random vector in R^d for the initial solution
    w1 = randn(d,1);
    % Generate a random real number for the initial bias
    b1 = randn;
    
    % Store in the matrix obj the values of the objective function during training
    [w,b,obj(i,:)] = train_logistic_regression_gd(Xtrain,ytrain,eta,T,w1,b1);
end

% It is fine to use plot(obj') too, but loglog makes the differences more evident
loglog(obj','LineWidth',2)
xlabel('Iterations')
ylabel('Objective Function')
grid on

% Save plot in a PNG file
print -dpng logistic_obj.png

%% 4.3d

% Close all plots
close all
% Clear the workspace
clear   

% Load training and test data
load adult_train_test

% Get number of training samples and their dimension
[m,d] = size(Xtrain);
% Get number of test samples and their dimension
[m2,d2]=size(Xtest);

eta = 1/5000;
T = 1000;
w1 = zeros(d,1);
b1 = 0;

[w,b,obj] = train_logistic_regression_gd(Xtrain,ytrain,eta,T,w1,b1);

% test your solution on the test set
mistakes = 0;
for i = 1:m2
    % Calculate prediction on test sample i and put it in hat_y
    % The following is equivalent to use the sigmoid and threshold to 0.5
    hat_y = sign(Xtest(i,:) * w + b);
    
    % Increase the number of mistakes if hat_y is different than ytest(i)
    mistakes = mistakes + (hat_y ~= ytest(i));
end
fprintf('%d mistakes over %d test samples\n', mistakes, m2);


