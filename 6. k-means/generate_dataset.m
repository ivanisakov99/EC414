function X = generate_dataset(N, mu, sigma)
[d, k] = size(mu);

y = randi([1, k], [N, 1]);
X = [];

for i = 1:k
    n_k = sum(y == i);
    mu_k = mu(:, i);
    sigma_k = sigma(i)*eye(d);
    X = [X; mvnrnd(mu_k, sigma_k, n_k)];
end 
end