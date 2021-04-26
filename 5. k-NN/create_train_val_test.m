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
for i =1:2:21
y_hat = predict_knn(Xtrain, ytrain, Xval, i);
acc(j) = sum((yval == y_hat)./length(yval));
j = j + 1;
end

figure;
plot(1:2:21, acc, 'LineWidth', 2);
hold on
xlabel('k', 'FontSize', 20);
ylabel('Accuracy', 'FontSize', 20);

[~, bestk] = max(acc);
k = (bestk * 2) - 1;
y_hat = predict_knn(Xtrain, ytrain, Xtest, k);
bestacc = sum((ytest == y_hat)./length(ytest));