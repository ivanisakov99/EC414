clear
close all

[X,y] = read_data; %Load Data

[m d] = size(X);
ntrain = round(m*0.60);
nval = round(m*0.20);
order = randperm(m);
Xtrain = X(order(1:ntrain),:);
ytrain = y(order(1:ntrain));
Xval = X(order(ntrain+1:ntrain+nval),:);
yval = y(order(ntrain+1:ntrain+nval));
Xtest = X(order(ntrain+nval+1:end),:);
ytest = y(order(ntrain+nval+1:end));

y_hat = kNN(Xval, Xtrain, ytrain, 3);

function y_hat = kNN(X_test, X, y, k)
[m1, d1] = size(X);
[m2, d2] = size(X_test);

X =reshape(X, 1, m1, d1);
Xtest = reshape(X_test, m2, 1, d2);
distance = sum((X - Xtest).^2, 3);
[~, nn_idx] = mink(distance, k, 2);
nn_y = y(nn_idx);
y_hat = mode(nn_y, 2);

end



% if nargin < 5
%     
%     X =reshape(X, 1, m1, d1);
%     Xtest = reshape(Xtest, m2, 1, d2);
%     distance = sum((X - Xtest).^2, 3);
%     [~, nn_idx] = mink(distance, k, 2);
%     nn_y = y(nn_idx);
%     yhat = mode(nn_y, 2);
%     return
% end
% 
% if(str == 'LOOCV')
%     X = reshape(X, 1, m1, d1);
%     Xtest = reshape(X, m1, 1, d1);
%     distance = sum((X - Xtest).^2, 3);
%     [~, nn_idx] = sort(distance, 2, 'ascend');
%     nn_idx = nn_idx(:, 2:end);
%     nn_y = y(nn_idx);
%     for i=1:length(nn_idx(1,:))/10
%         yhat(:, i) = mode(nn_y(:, [1:i*10]), 2);
%     end
% end
% end