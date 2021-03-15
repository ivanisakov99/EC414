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

j = 1;
for i =1:2:15
y_hat = predict_knn(Xtrain, ytrain, Xval, i);
acc(j) = sum((yval == y_hat)./length(yval));
j = j + 1;
end
figure;
plot(1:2:15, acc, 'LineWidth', 2);
hold on
xlabel('k', 'FontSize', 20);
ylabel('Accuracy', 'FontSize', 20);

function [yhat] = predict_knn(X, y, Xtest, k)
ed = zeros(size(Xtest,1), size(X,1)); 
ind = zeros(size(Xtest,1), size(X,1));

for test_point=1:size(Xtest,1)
    for train_point=1:size(X,1)
        ed(test_point,train_point)=sqrt(sum((Xtest(test_point,:) - X(train_point,:)).^2));
    end
    [~,ind(test_point,:)]=sort(ed(test_point,:));
end

k_nn=ind(:,1:k);
nn_y = y(k_nn);
yhat = mode(nn_y, 2);
end

