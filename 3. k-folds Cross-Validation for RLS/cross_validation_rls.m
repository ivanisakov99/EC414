function [validation_loss] = cross_validation_rls(X, y, lambda, k)
[m,d]=size(X);
fold_length = m/k;
validation_loss = 0;

% Divide training data in n_folds folds and for each fold train and record
% validation error. Return the average of the validation losses over the
% folds
% The loss is the square loss

    for i=1:k
        % create validation indexes for fold k
        %val_idx;
        X_test = X((i-1)*fold_length + 1: i*fold_length,:);
        y_test = y((i-1)*fold_length + 1: i*fold_length);
       
        % create training indexes for fold k
        %train_idx;
        X_train1 = X(1:(i-1)*fold_length,:);
        X_train2 = X(i*fold_length +1: end, :);
        X_train = [X_train1; X_train2];
       
        
        y_train1 = y(1:(i-1)*fold_length);
        y_train2 = y(i*fold_length +1: end);
        y_train = [y_train1; y_train2]; 

        
        clear X_train1 X_train2;

        % train on the n_folds-1 training folds using RLS 
        [w,b]=train_rls(X_train,y_train,lambda);
        % test on the validation fold
        y_hat = X_test * w + b;
        validation_loss = validation_loss + ((y_test - y_hat)'*(y_test - y_hat))/length(y_test);
    end
validation_loss = validation_loss/k;
end