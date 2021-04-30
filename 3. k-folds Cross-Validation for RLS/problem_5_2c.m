%% Problem 5.2: k-folds cross-validation for Regularized Least Square Regression with polynomials

clear all
close all

rand('state',0)

n_folds = 10;
poly_degree = 10;
lambda = 1e-3;

load cadata_train_test

[m,d] = size(Xtrain_norm);

% Shuffle training input and labels
idx = randperm(m);
Xtrain_norm = Xtrain_norm(idx,:);
ytrain = ytrain(idx);

% Generate poly features
[Xtrain_poly] = generate_poly_features(Xtrain_norm,poly_degree);
[Xtest_poly] = generate_poly_features(Xtest_norm,poly_degree);

% Divide training data in 10 folds and for each fold train and record
% validation error for each value of the degree of the fractional polynomial.
% The loss is the square loss
val_loss_per_hyperparam = zeros(poly_degree,1);
for j = 1:poly_degree
    val_loss_per_hyperparam(j) = cross_validation_rls(Xtrain_poly(:,1:j*d),ytrain,lambda,n_folds);
end
    
plot(val_loss_per_hyperparam,'r')
xlabel('max fractional poly degree')
ylabel('Errors')
legend('10-folds Cross Validation Loss')

% Best degree
[~,best_degree]=min(val_loss_per_hyperparam);
fprintf('The degree of the fractional polynomial that gives the best result is %d\n', best_degree);

% Retrain on everything with the selected degree
[w,b]=train_rls(Xtrain_poly(:,1:best_degree*d),ytrain,1e-3);
test_loss=mean((Xtest_poly(:,1:best_degree*d)*w+b-ytest).^2);
fprintf('The test loss of your predictor is %d\n', test_loss);
