function yhat = predict_knn(X,y,Xtest,k)

[n d] = size(Xtest);
[m d] = size(X);
yhat = zeros(n,1);

for i = 1:n
    Xtemp = Xtest(i,:);
    distances = sqrt(sum((X - repmat(Xtemp,m,1)).^2,2));
    [sorteddist sortedind] = sort(distances);
    yhat(i) = sign(sum(y(sortedind(1:k))));
end
