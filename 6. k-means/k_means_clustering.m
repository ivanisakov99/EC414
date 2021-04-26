function [c, obj, y] = k_means_clustering(X, c0, T)
[m, d] = size(X);
[k, d] = size(c0);

y_hat = 0;
distance = zeros(m, k);
for j = 1:T   
    
    for l = 1:k
        distance(:,l) = sum((X - c0(l,:)).^2, 2);
    end

    
    [objf, y_hat] = min(distance, [], 2);
    
    obj = sum(objf);
    
    for i = 1:k
        index = find(y_hat == i);
        c0(i, :) = mean(X(index, :));
    end
end

c = c0;
y = y_hat;

end

