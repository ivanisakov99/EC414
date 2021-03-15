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