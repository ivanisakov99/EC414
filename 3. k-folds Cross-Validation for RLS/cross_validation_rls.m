function [validation_loss] = cross_validation(X, y, lambda, n_folds)
[m,d]=size(X);

% Divide training data in n_folds folds and for each fold train and record
% validation error. Return the average of the validation losses over the
% folds
% The loss is the square loss

validation_loss=0;
for k=1:n_folds
    % create validation indexes for fold k
    val_idx=zeros(m,1,'logical');
    val_idx(k:n_folds:m)=1;

    % samples not in the validation are in the training
    train_idx=zeros(m,1,'logical');
    train_idx(val_idx==0)=1;

    % train on the n_folds-1 training folds
    [w,b]=train_rls(X(train_idx,:),y(train_idx),lambda);
    % test on the validation fold
    validation_loss=validation_loss+mean((X(val_idx,:)*w+b-y(val_idx)).^2);
end
validation_loss=validation_loss/n_folds;
