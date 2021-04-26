clear
close all

rand('state',0)

load cadata_train_test

n_folds = 8;
poly_degree = 10;
lambda = 0.001;

[m,d]=size(Xtrain_norm);

validation_loss = zeros(poly_degree,1);
mse_test = zeros(poly_degree, 1);

%shuffle training input and labels
idx=randperm(m);
Xtrain_norm=Xtrain_norm(idx,:);
ytrain=ytrain(idx);

% generate the polynomial features calling generate_poly_features
X_poly = generate_poly_features(Xtrain_norm, poly_degree);

X_poly_test = generate_poly_features(Xtest_norm, poly_degree);


% Call cross_validation_rls for each degree of the polynomial to try and record the validation loss
for i=1:poly_degree
    validation_loss(i) = cross_validation_rls(X_poly(:, 1:8*i), ytrain, lambda, n_folds);
end

figure(1)
plot(validation_loss, 'LineWidth', 2)
xlabel('Polynomial Degree', 'FontSize', 20)
title('10-fold Cross Validation')

%5.2e
best_poly = 3;



X_poly_best = generate_poly_features(Xtrain_norm, best_poly);
X_poly_test = generate_poly_features(Xtest_norm, best_poly);
[wtilde, btilde] = train_rls(X_poly_best, ytrain, lambda);
y_hat = X_poly_test*wtilde + btilde;
mse_test = mean((y_hat - ytest)).^2


