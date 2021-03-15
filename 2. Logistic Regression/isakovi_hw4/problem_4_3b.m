% close all plots
close all 
% clear the workspace
clear   

% load training and test data
load adult_train_test

% Get number of training samples and their dimension
[m,d]=size(Xtrain);

eta = 1/5000;
T = 1000;
obj = zeros(10, 1000);
for i=1:10
    % Generate a random vector in R^d for the initial solution
    w1 = randn(d,1);
    % Generate a random real number for the initial bias
    b1 = randn(1,1);
    
    % store in the matrix obj the values of the objective function during training
    [w,b,obj(i,:)] = train_logistic_regression_gd(Xtrain,ytrain,eta,T,w1,b1);
end

% Plot the 10 different lines for the objective function 
loglog(obj');

% Save plot in a PNG file
print -dpng logistic_obj.png
