function [y_hat, c] = kmeans(X, k, sc)
[m, d] = size(X);
c = sc * (rand(k, d));

X = reshape(X, [m, 1, d]);
c = reshape(c, [1, k, d]);

y_hat = 0;
y_hatp = 1;

while(sum(y_hatp ~= y_hat) > 0)
    y_hatp = y_hat;
    
    distance = sum((X - c) .^2, 3);
    [~, y_hat] = min(distance, [], 2);
    
    for i = 1:k
        index = find(y_hat == i);
        c(1, i, :) = squeeze( mean(X(index, 1, :) ));
    end
end

c = squeeze(c);
end